#!/usr/bin/env python3
"""
Full evaluation pipeline for PA-BPE / MYTE / BLT tokenizer comparison on FLORES+.
Metrics: BLEU (OpenSEAL tokenizer), spBLEU (flores200 SentencePiece), chrF++ (word_order=2)


BLT uses a completely separate inference path via the bytelatent library:
  - load_consolidated_model_and_tokenizer  (not HuggingFace AutoModel)
  - generate_nocache                       (not model.generate)
  - BltTokenizer.decode                   (not HuggingFace tokenizer)
  - A patcher built from train_cfg with realtime_patching=True


Usage:
    # PA-BPE or MYTE (standard HuggingFace path)
    python eval_pipeline.py \
        --model_path /path/to/olmo-pabpe \
        --tokenizer_name pabpe \
        --flores_dir /scratch/.../flores-plus_dev_devtest


    # BLT (bytelatent path)
    python eval_pipeline.py \
        --model_path /path/to/hf-weights/blt_7b \
        --tokenizer_name blt \
        --entropy_model_path /path/to/hf-weights/entropy_model \
        --flores_dir /scratch/.../flores-plus_dev_devtest
"""


import argparse
import json
import re
import signal
import csv
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo
import sacrebleu
from transformers import AutoTokenizer, AutoModelForCausalLM

from pythainlp.tokenize import word_tokenize as thai_word_tokenize
from indicnlp.tokenize import indic_tokenize
from khmernltk import word_tokenize as khmer_word_tokenize
from laonlp.tokenize import word_tokenize as lo_word_tokenize
from myTokenize import WordTokenizer as myWordTokenizer

import sys
sys.path.insert(0, "/scratch/Projects/CFP-01/CFP01-CF-060/kieron/blt")

from bytelatent.generate_blt import generate_nocache
from bytelatent.distributed import DistributedArgs, setup_torch_distributed
from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.blt_tokenizers.blt_tokenizer import BltTokenizer


# ---------------------------------------------------------------------------
# 1.  FLORES+ LOCAL FILE MAPPING
# ---------------------------------------------------------------------------


FLORES_FILENAME = {
    "en": "eng_Latn",
    "id": "ind_Latn",
    "km": "khm_Khmr",
    "lo": "lao_Laoo",
    "ms": "zsm_Latn",
    "my": "mya_Mymr",
    "ta": "tam_Taml",
    "th": "tha_Thai",
    "tl": "fil_Latn",
    "vi": "vie_Latn",
    "zh": "cmn_Hans",
}


SEA_LANGS = ["id", "km", "lo", "ms", "my", "ta", "th", "tl", "vi", "zh"]



def load_flores_local(src_lang: str, tgt_lang: str, flores_dir: str | Path) -> tuple[list[str], list[str]]:
    """Load FLORES+ devtest sentences from local .devtest files."""
    flores_dir = Path(flores_dir)
    src_file = flores_dir / f"{FLORES_FILENAME[src_lang]}.devtest"
    tgt_file = flores_dir / f"{FLORES_FILENAME[tgt_lang]}.devtest"


    if not src_file.exists():
        raise FileNotFoundError(f"FLORES+ file not found: {src_file}")
    if not tgt_file.exists():
        raise FileNotFoundError(f"FLORES+ file not found: {tgt_file}")


    src_sents = [s for s in src_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    tgt_sents = [s for s in tgt_file.read_text(encoding="utf-8").splitlines() if s.strip()]


    assert len(src_sents) == len(tgt_sents), (
        f"Line count mismatch: {src_file} ({len(src_sents)}) vs {tgt_file} ({len(tgt_sents)})"
    )
    return src_sents, tgt_sents



# ---------------------------------------------------------------------------
# 2.  PROMPT CONFIGS
# ---------------------------------------------------------------------------


_SHOT_EN = [
    "The War of Spanish Succession marked the first war whose central issue was the balance of power.",
    "Many entire nations are completely fluent in English, and in even more you can expect a limited knowledge - especially among younger people.",
    "Fewer than a thousand cases have ever been reported in humans, but some of them have been fatal.",
    "In the last 3 months, over 80 arrestees were released from the Central Booking facility without being formally charged.",
    "The focus of this mindset is speed, logic and accuracy, also identification of facts, reapplying existing techniques, gathering information.",
]


_SHOT_EN_TL = [
    "They include the Netherlands, with Anna Jochemsen finishing ninth in the women's standing class in the Super-G yesterday, and Finland with Katja Saarinen finishing tenth in the same event.",
    "So many of us find ourselves watching a television show that informs us of a process or experience in which we will never participate or apply that knowledge.",
    "He arrived in the US with 4 cents to his name, a book of poetry, and a letter of recommendation from Charles Batchelor (his manager in his previous job) to Thomas Edison.",
    "The great pyramid was created to honor the Pharaoh Khufu, and many of the smaller pyramids, tombs, and temples were built to honor Khufu's wives and family members.",
    "Tours are cheaper for larger groups, so if you're by yourself or with just one friend, try to meet other people and form a group of four to six for a better per-person rate.",
]


_SHOT_TGT = {
    "id": [
        "Perang Penerus Spanyol menandai perang pertama dengan isu sentral berupa keseimbangan kekuatan.",
        "Banyak negara yang sepenuhnya fasih dalam bahasa Inggris, bahkan lebih banyak yang memiliki pengetahuan terbatas dari yang dapat Anda perkirakan - terutama di kalangan anak muda.",
        "Kurang dari seribu kasus pernah dilaporkan terjadi pada manusia, tetapi beberapa di antaranya merupakan kasus yang fatal.",
        "Dalam 3 bulan terakhir, lebih dari 80 tahanan dibebaskan dari fasilitas Central Booking tanpa dikenakan biaya secara resmi.",
        "Fokus dari pola pikir ini adalah kecepatan, logika, dan ketepatan, juga identifikasi fakta, penerapan kembali teknik yang ada, pengumpulan informasi.",
    ],
    "km": [
        "សង្គ្រាមដណ្តើមអំណាចគ្នារបស់ប្រទេសអេស្ប៉ាញ គឺជាសង្គ្រាមដំបូងគេ ដែល​បញ្ហាស្នូលនៃ​សង្គ្រាម​នេះ​ គឺ​តុល្យភាព​អំណាច។",
        "មានប្រជា​ជាតិ​ជាច្រើន​ដែល​ចេះ​ប្រើ​ភាសាអង់គ្លេស​បាន​ល្អ​ ហើយ​ជាង​នេះ​ទៅ​ទៀត​ ​អ្នក​អាចរំពឹង​ថា ពួកគេមានចំណេះមាន​កម្រិត​ ជាពិសេស​ក្នុង​ចំណោមមនុស្សវ័យក្មេង។",
        "បានរាយការណ៍ថា​មាន​តិចជាងមួយពាន់ករណី​បានកើតឡើង​ជាមួយនឹងមនុស្ស ប៉ុន្តែពួកគេខ្លះបានស្លាប់បាត់ទៅហើយ។",
        "ក្នុងរយៈពេល 3 ខែ ចុងក្រោយ ជន​ត្រូវ​ចាប់ខ្លួន​ជាង 80 នាក់ ត្រូវ​បាន​ដោះលែង​ចេញ​ពី​មជ្ឈមណ្ឌល​ចុះ​បញ្ជី​កណ្តាល​ដោយ​មិន​បាន​ចោទ​ប្រកាន់​ជាផ្លូវការ​ឡើយ។",
        "ការផ្តោតលើផ្នត់គំនិតនេះ គឺល្បឿន តក្កវិជ្ជា និងភាពត្រឹមត្រូវ ព្រមទាំងការកំណត់ការពិត ការអនុវត្តបច្ចេកទេសដែលមានស្រាប់ឡើងវិញ ការប្រមូលព័ត៌មាន។",
    ],
    "lo": [
        "ສົງຄາມແຫ່ງຄວາມສຳເລັດຂອງແອັດສະປາຍໄດ້ເຮັດສົງຄາມຄັ້ງທຳອິດເຊິ່ງບັນຫາໃຈກາງແມ່ນຄວາມສົມດຸນຂອງອຳນາດ.",
        "ທັງໝົດປະເທດຂອງຫຼາຍໆປະເທດແມ່ນມີຄວາມຄ່ອງແຄ້ວໃນການໃຊ້ພາສາອັງກິດ ແລະ ມັນຫຼາຍກວ່າທີ່ທ່ານຈະຄາດຫວັງເຖິງຄວາມຮູ້ທີ່ຈຳກັດ - ໂດຍສະເພາະໃນກຸ່ມໄວໜຸ່ມ.",
        "ມີຫນ້ອຍກວ່າພັນກໍລະນີທີ່ເຄີຍຖືກລາຍງານມາກ່ຽວກັບມະນຸດ, ແຕ່ວ່າມີບາງກໍລະນີກໍເປັນໂຣກຮ້າຍແຮງ.",
        "ໃນ 3 ເດືອນຜ່ານມາ, ຜູ້ຈັບກຸມຫຼາຍກວ່າ 80 ຄົນໄດ້ຖືກປ່ອຍຕົວອອກຈາກສູນກັກກັນໂດຍບໍ່ມີຂໍ້ກ່າວຫາຢ່າງເປັນທາງການ.",
        "ຈຸດລວມສຸມຂອງທັດສະນະຄະຕິນີ້ແມ່ນ ຄວາມໄວ, ໂລຊິກ ແລະ ຄວາມຖືກຕ້ອງແມ່ນຢຳ, ພ້ອມທັງການກຳນົດຂໍ້ເທັດຈິງ, ການນຳເຕັກນິກທີ່ມີຢູ່ມາໝູນໃຊ້ໃໝ່, ການລວບລວມຂໍ້ມູນ.",
    ],
    "ms": [
        "Perang Pewarisan Sepanyol menandakan perang pertama yang mana keseimbangan kuasa merupakan persoalan pertama.",
        "Banyak negara petah dalam bahasa Inggeris, dan dalam lebih banyak negara anda boleh menjangkakan pengetahuan yang berbatas - terutamanya dalam khalayak orang muda.",
        "Kurang daripada seribu kes yang pernah dilaporkan pada manusia, tetapi beberapa antaranya membawa maut.",
        "Dalam 3 bukan yang lepas, lebih 80 orang yang ditangkap telah dibebaskan daripada fasiliti Central Booking tanpa dicaj secara formal.",
        "Fokus untuk set pemikiran ini adalah kelajuan, logik, dan ketepatan, serta pengenalpastian fakta, menggunakan semua teknik yang ada, serta mengumpul maklumat.",
    ],
    "my": [
        "စပိန်တို့၏စစ်ပွဲအောင်မြင်မှုသည် အဓိကပြဿနာမှာ အာဏာ၏ဟန်ချက်ညီမှုဖြစ်ခဲ့သော ပထမအကြိမ်စစ်ပွဲကို အမှတ်အသားဖြစ်စေခဲ့ပါသည်။",
        "နိုင်ငံသားအများစုက အင်္ဂလိပ်လို ကောင်းစွာ ပြောနိုင်ပြီး အထူးသဖြင့် လူငယ်များသည် သင်ထင်ထားသည်ထက်ကို ပို၍ ဗဟုသုတ ရှိကြပါသည်။",
        "လူတစ်ထောင်ခန့် ဖြစ်ပွားကြောင်း အစီရင်ခံထားသော်လည်း ၎င်းတို့အနက်အချို့မှာ သေစေနိုင်ပါသည်။",
        "လွန်ခဲ့သော ၃ လတွင်၊ ဖမ်းဆီးခံရသူပေါင်း ၈၀ ကျော်ကို တရားဝင်တရားစွဲဆိုခြင်းမရှိဘဲ Central Booking အဆောက်အအုံမှ လွှတ်ပေးခဲ့ပါသည်။",
        "ဤစိတ်သဘောထား၏ အာရုံစိုက်မှုမှာ လက်ရှိနည်းလမ်းများကို ပြန်လည်အသုံးချရင်း အချက်အလက်များ စုဆောင်းကာ မြန်နှုန်း၊ ယုတ္တိဗေဒနှင့် တိကျမှုတို့အပြင် အချက်လက်များ ဖော်ထုတ်ခြင်း ဖြစ်သည်။",
    ],
    "ta": [
        "அதிகார சமநிலைதான் ஸ்பானிஷ் வாரிசுப் போருக்கான முதல் போரின் முக்கிய பிரச்சனையாக இருந்தது.",
        "பல தேசங்கள் முழுவதுமே ஆங்கிலத்தில் சரளமாக உள்ளன மற்றும் மேலும் பலவற்றில் குறைந்த அளவு ஆங்கில அறிவை, குறிப்பாக இளைஞர்களிடத்தில், எதிர் பார்க்கலாம்.",
        "மனிதர்களில் ஆயிரத்திற்கும் குறைவான வழக்குகள் அறிவிக்கப்பட்டுள்ளன, ஆனால், அவற்றில் சில உயிரைப் போக்கியவை.",
        "கடந்த 3 மாதங்களில், 80-க்கும் மேற்பட்ட கைது செய்யப்பட்டவர்கள், முறையாகக் குற்றஞ்சாட்டு தாக்கல் செய்யப்படாமல், மத்திய முன்பதிவு நிலையத்திலிருந்து விடுவிக்கப்பட்டனர்.",
        "இந்த மனப்பாங்கின் கவனம், வேகம், லாஜிக் மற்றும் துல்லியம் ஆகியவை, மேலும் உண்மைகளின் கண்டுபிடிப்பு, இருக்கும் உத்திகளின் மறு பயன்பாடு, தகவல் சேகரித்தல்.",
    ],
    "th": [
        "สงครามสืบราชบัลลังก์สเปนนับเป็นสงครามครั้งแรกซึ่งปัญหาหลักคือเรื่องดุลอำนาจ",
        "หลายประเทศที่ประชากรทุกคนใช้ภาษาอังกฤษได้อย่างคล่องแคล่วทั่วทั้งประเทศ และยิ่งไปกว่านั้นคุณคาดการณ์ได้เลยว่าจะพบกับคนที่มีความรู้จำกัด โดยเฉพาะในกลุ่มประชากรที่มีอายุน้อยกว่า",
        "มีรายงานผู้ป่วยน้อยกว่าหนึ่งพันรายในมนุษย์ แต่บางรายมีความรุนแรงถึงขั้นทำให้เสียชีวิตได้",
        "ในช่วง 3 เดือนที่ผ่านมา ผู้ถูกจับกุมกว่า 80 รายได้รับการปล่อยตัวจากศูนย์บันทึกข้อหากลางโดยไม่ได้ถูกตั้งข้อหาอย่างเป็นทางการ",
        "หลักสำคัญของวิธีคิดนี้คือ ความรวดเร็ว ตรรกะ และความแม่นยำ รวมทั้งการบ่งบอกข้อเท็จจริง การนำเทคนิคที่มีอยู่แล้วมาใช้ซ้ำ และการรวบรวมข้อมูล",
    ],
    "tl": [
        "Kasama dito ang Netherlands, dahil si Anna Jochemsen ay nagtapos na ikasiyam sa klaseng pambabaeng nakatayo sa Super-G kahapon, at ang Finland dahil si Katja Saarinen ay nagtapos na ikasampu sa parehong laban.",
        "Napakarami sa atin ang nasusumpungan ang ating mga sarili na nanonood ng palabas sa telebisyon na nagbibigay-kaalaman sa atin tungkol sa isang proseso o karanasan kung saan hindi kailanman tayo makikibahagi o gagamit ng kaalamang iyon.",
        "Dumating siya sa US na may 4 na sentimo sa kaniyang pangalan, isang libro ng mga tula, at isang sulat ng rekomendasyon mula kay Charles Batchelor (ang kaniyang tagapamahala sa dati niyang trabaho) patungo kay Thomas Edison.",
        "Ang great pyramid ay ginawa upang parangalan ang Pharaoh na si Khufu, at marami sa mga maliliit na pyramid, mga puntod, at mga templo ay ginawa upang parangalan ang mga asawa ni Khufu at mga miyembro ng pamilya.",
        "Ang mga paglilibot ay mas mura para sa mas malalaking grupo, kaya kung ikaw ay mag-isa o may iisang kasama, iyong subukang makatagpo ng ibang tao at bumuo ng grupo ng apat hanggang anim upang makakuha ng mas mababang singil para sa bawat tao.",
    ],
    "vi": [
        "Chiến tranh Kế vị Tây Ban Nha đã đánh dấu chiến tranh đầu tiên mà vấn đề trọng tâm là sự cân bằng quyền lực.",
        "Nhiều quốc gia hoàn toàn thông thạo tiếng Anh, và ở nhiều quốc gia khác người dân cũng hiểu biết phần nào - nhất là trong số những người trẻ tuổi.",
        "Chỉ có chưa tới một ngàn ca bệnh ở người được báo cáo, nhưng một số ca đã dẫn đến tử vong.",
        "Trong vòng 3 tháng qua, đã có hơn 80 người bị bắt được thả ra khỏi trụ sở của Central Booking và không bị buộc tội chính thức.",
        "Sự tập trung tâm trí là tốc độ, sự hợp lý và tính chính xác, cũng như sự xác định thực tế, áp dụng lại các kỹ thuật có sẵn, thu thập thông tin.",
    ],
    "zh": [
        "西班牙继承权之战标志着第一场以权力平衡为核心问题的战争。",
        "在许多国家/地区，全国的人都能说一口流利的英语，而在更多的国家/地区，人们对英语也略知一二，尤其是年轻人。",
        "据报道，人类感染禽流感的病例不到 一千例，但其中有一些病例是致命的。",
        "在过去的 3 个月里，超过 80 名被捕者在没有被正式起诉的情况下从中央拘留所释放。",
        "这种思维模式看重的是速度、逻辑和准确性，还有甄别事实、对现有技术的重新应用和信息收集。",
    ],
}


LANG_PAIR_META = {
    ("en", "id"): dict(instruction="Terjemahkan teks berikut ini ke dalam bahasa Indonesia.\n\nJawablah hanya dengan menggunakan format berikut ini:\nTerjemahan: $TRANSLATION\nGanti $TRANSLATION dengan teks yang telah diterjemahkan.", src_label="Teks", tgt_label="Terjemahan"),
    ("en", "km"): dict(instruction="បកប្រែអត្ថបទខាងក្រោមទៅជាភាសាខ្មែរ។\n\nឆ្លើយតបតែក្នុងទម្រង់ខាងក្រោម:\nការបកប្រែ: $TRANSLATION\nជំនួស $TRANSLATION ជាមួយអត្ថបទដែលបានបកប្រែ។", src_label="អត្ថបទ", tgt_label="ការប្រែសម្រួល"),
    ("en", "lo"): dict(instruction="ແປຂໍ້ຄວາມຂ້າງລຸ່ມເປັນພາສາລາວ.\n\nຕອບກັບໃນຮູບແບບຕໍ່ໄປນີ້:\nການແປ: $TRANSLATION\nແທນທີ່ $TRANSLATION ດ້ວຍຂໍ້ຄວາມທີ່ແປແລ້ວ.", src_label="ຂໍ້ຄວາມ", tgt_label="ການແປ"),
    ("en", "ms"): dict(instruction="Terjemahkan teks di bawah ke dalam bahasa Melayu.\n\nBalas hanya dalam format berikut:\nTerjemahan: $TRANSLATION\nGanti $TRANSLATION dengan teks terjemahan.", src_label="Teks", tgt_label="Terjemahan"),
    ("en", "my"): dict(instruction="အောက်ဖော်ပြပါစာသားကို မြန်မာဘာသာသို့ ဘာသာပြန်ပါ။\n\nအောက်ပါဖော်မက်ဖြင့်သာ စာပြန်ပါ-\nဘာသာပြန်ချက်- $TRANSLATION\nဘာသာပြန်ထားသော စာသားဖြင့် $TRANSLATION ကို အစားထိုးပါ။", src_label="စာသား", tgt_label="ဘာသာပြန်ချက်"),
    ("en", "ta"): dict(instruction="பின்வரும் உரையைத் தமிழ் மொழிக்கு மொழிபெயர்க்கவும்.\n\nபின்வரும் பதில் வடிவமைப்பை மட்டும் பயன்படுத்தி பதிலளிக்கவும்:\nமொழிபெயர்ப்பு: $TRANSLATION\nமொழிபெயர்த்த உரையுடன் $TRANSLATION ஐ மாற்றவும்.", src_label="உரை", tgt_label="மொழிபெயர்ப்பு"),
    ("en", "th"): dict(instruction="แปลข้อความต่อไปนี้เป็นภาษาไทย\n\nจงตอบตามรูปแบบดังต่อไปนี้:\nคำแปล: $TRANSLATION\nโดยจะต้องแทนที่ $TRANSLATION ด้วยข้อความที่แปลแล้ว", src_label="ข้อความ", tgt_label="คำแปล"),
    ("en", "tl"): dict(instruction="Isalin ang sumusunod na teksto sa Filipino.\n\nTumugon gamit ang sumusunod na format:\nSalin: $TRANSLATION\nPalitan ang $TRANSLATION gamit ng isinalin na teksto.", src_label="Teksto", tgt_label="Salin"),
    ("en", "vi"): dict(instruction="Dịch văn bản dưới đây sang Tiếng Việt.\n\nChỉ trả lời bằng cách sử dụng định dạng sau:\nBản dịch: $TRANSLATION\nThay thế $TRANSLATION bằng văn bản đã dịch.", src_label="Văn bản", tgt_label="Bản dịch"),
    ("en", "zh"): dict(instruction="将以下文本翻译成中文。\n\n请仅按以下格式回复:\n翻译: $TRANSLATION\n将 $TRANSLATION 替换为翻译后的文本。", src_label="文本", tgt_label="翻译"),
    ("id", "en"): dict(instruction="Terjemahkan teks berikut ini ke dalam bahasa Inggris.\n\nJawablah hanya dengan menggunakan format berikut ini:\nTerjemahan: $TRANSLATION\nGanti $TRANSLATION dengan teks yang telah diterjemahkan.", src_label="Teks", tgt_label="Terjemahan"),
    ("km", "en"): dict(instruction="បកប្រែអត្ថបទខាងក្រោមទៅជាភាសាអង់គ្លេស\n\nឆ្លើយតបតែក្នុងទម្រង់ខាងក្រោម:\nការបកប្រែ: $TRANSLATION\nជំនួស $TRANSLATION ជាមួយអត្ថបទដែលបានបកប្រែ។", src_label="អត្ថបទ", tgt_label="ការប្រែសម្រួល"),
    ("lo", "en"): dict(instruction="ແປຂໍ້ຄວາມຕໍ່ໄປນີ້ເປັນພາສາອັງກິດ\n\nຕອບກັບໃນຮູບແບບຕໍ່ໄປນີ້:\nການແປ: $TRANSLATION\nແທນທີ່ $TRANSLATION ດ້ວຍຂໍ້ຄວາມທີ່ແປແລ້ວ.", src_label="ຂໍ້ຄວາມ", tgt_label="ການແປ"),
    ("ms", "en"): dict(instruction="Terjemahkan teks berikut ke dalam bahasa Inggeris.\n\nBalas hanya dalam format berikut:\nTerjemahan: $TRANSLATION\nGanti $TRANSLATION dengan teks terjemahan.", src_label="Teks", tgt_label="Terjemahan"),
    ("my", "en"): dict(instruction="အောက်ဖော်ပြပါစာသားကို အင်္ဂလိပ်ဘာသာသို့ ဘာသာပြန်ပါ။\n\nအောက်ပါဖော်မက်ဖြင့်သာ စာပြန်ပါ-\nဘာသာပြန်ချက်- $TRANSLATION\nဘာသာပြန်ထားသော စာသားဖြင့် $TRANSLATION ကို အစားထိုးပါ။", src_label="စာသား", tgt_label="ဘာသာပြန်ချက်"),
    ("ta", "en"): dict(instruction="பின்வரும் உரையை ஆங்கிலத்தில் மொழிபெயர்க்கவும்.\n\nபின்வரும் பதில் வடிவமைப்பை மட்டும் பயன்படுத்தி பதிலளிக்கவும்:\nமொழிபெயர்ப்பு: $TRANSLATION\nமொழிபெயர்த்த உரையுடன் $TRANSLATION ஐ மாற்றவும்.", src_label="உரை", tgt_label="மொழிபெயர்ப்பு"),
    ("th", "en"): dict(instruction="แปลข้อความต่อไปนี้เป็นภาษาอังกฤษ\n\nจงตอบตามรูปแบบดังต่อไปนี้:\nคำแปล: $TRANSLATION\nโดยจะต้องแทนที่ $TRANSLATION ด้วยข้อความที่แปลแล้ว", src_label="ข้อความ", tgt_label="คำแปล"),
    ("tl", "en"): dict(instruction="Isalin ang sumusunod na teksto sa English.\n\nTumugon gamit ang sumusunod na format:\nSalin: $TRANSLATION\nPalitan ang $TRANSLATION gamit ng isinalin na teksto.", src_label="Teksto", tgt_label="Salin"),
    ("vi", "en"): dict(instruction="Dịch văn bản dưới đây sang Tiếng Anh.\n\nChỉ trả lời bằng cách sử dụng định dạng sau:\nBản dịch: $TRANSLATION\nThay thế $TRANSLATION bằng văn bản đã dịch.", src_label="Văn bản", tgt_label="Bản dịch"),
    ("zh", "en"): dict(instruction="将以下文字翻译成英文\n\n请仅按以下格式回复:\n翻译: $TRANSLATION\n将 $TRANSLATION 替换为翻译后的文本。", src_label="文本", tgt_label="翻译"),
}



# ---------------------------------------------------------------------------
# 3.  PROMPT BUILDING
# ---------------------------------------------------------------------------


def _format_shot(src_label, tgt_label, src_text, tgt_text):
    return f"{src_label}:\n```\n{src_text}\n```\n{tgt_label}: {tgt_text}"



def build_prompt(src_lang: str, tgt_lang: str, test_src: str) -> str:
    meta = LANG_PAIR_META[(src_lang, tgt_lang)]
    src_label, tgt_label = meta["src_label"], meta["tgt_label"]


    if src_lang == "en":
        src_shots = _SHOT_EN_TL if tgt_lang == "tl" else _SHOT_EN
        tgt_shots = _SHOT_TGT[tgt_lang]
    else:
        tgt_shots = _SHOT_EN_TL if src_lang == "tl" else _SHOT_EN
        src_shots = _SHOT_TGT[src_lang]


    shots = "\n\n".join(_format_shot(src_label, tgt_label, s, t) for s, t in zip(src_shots, tgt_shots))
    return f"{meta['instruction']}\n\n{shots}\n\n{src_label}:\n```\n{test_src}\n```\n{tgt_label}:"



# ---------------------------------------------------------------------------
# 4.  LANGUAGE-SPECIFIC PRE-TOKENIZATION  (based on tokenize_text.py)
# ---------------------------------------------------------------------------


PRETOK_LANGS = {"th", "km", "lo", "my", "ta"}
TIMEOUT_SEC  = 1



class TimeoutError_(Exception):
    pass



def _alarm_handler(signum, frame):
    raise TimeoutError_()



signal.signal(signal.SIGALRM, _alarm_handler)



def pretokenize_line(line: str, lang: str) -> str:
    lang = lang.lower()
    if lang == "th":
        return " ".join(thai_word_tokenize(line, engine="newmm"))
    elif lang == "ta":
        return " ".join(indic_tokenize.trivial_tokenize(line, lang="ta"))
    elif lang == "km":
        return " ".join(khmer_word_tokenize(line, return_tokens=True))
    elif lang == "lo":
        signal.alarm(TIMEOUT_SEC)
        try:
            return " ".join(lo_word_tokenize(line))
        except TimeoutError_:
            return line
        finally:
            signal.alarm(0)
    elif lang == "my":
        return " ".join(myWordTokenizer().tokenize(line))
    else:
        raise ValueError(f"pretokenize_line called on non-pretok lang: {lang}")



def pretokenize_corpus(lines: list[str], lang: str) -> list[str]:
    return [pretokenize_line(line, lang) for line in lines]



# ---------------------------------------------------------------------------
# 5.  METRICS: BLEU + spBLEU + chrF++
# ---------------------------------------------------------------------------


def compute_bleu(hyps_tok: list[str], refs_tok: list[str], tgt_lang: str) -> float:
    if tgt_lang == "en":
        metric = sacrebleu.BLEU(tokenize="13a")
    elif tgt_lang == "zh":
        metric = sacrebleu.BLEU(tokenize="zh")
    elif tgt_lang in PRETOK_LANGS:
        metric = sacrebleu.BLEU(tokenize="none")
    else:
        metric = sacrebleu.BLEU(tokenize="intl")
    return metric.corpus_score(hyps_tok, [refs_tok]).score



def compute_spbleu(hyps_raw: list[str], refs_raw: list[str]) -> float:
    return sacrebleu.BLEU(tokenize="flores200").corpus_score(hyps_raw, [refs_raw]).score



def compute_chrf(hyps_raw: list[str], refs_raw: list[str]) -> float:
    return sacrebleu.CHRF(word_order=2).corpus_score(hyps_raw, [refs_raw]).score


# ---------------------------------------------------------------------------
# 6.  OUTPUT EXTRACTION  (shared between HF and BLT paths)
# ---------------------------------------------------------------------------

def extract_translation(raw_output: str, tgt_label: str) -> str:
    """
    Strip the target-language label prefix and take the first line.
    raw_output must be the GENERATED text only (prompt already stripped).
    """
    text = raw_output.strip()
    if text.startswith(tgt_label + ":"):
        text = text[len(tgt_label) + 1:].strip()
    text = text.replace("\n", " ").strip()
    text = re.sub(r"^\s*-\s*", "", text)
    return text

'''
def extract_translation(raw_output: str, tgt_label: str) -> str:
    text = raw_output.strip()
    # More robust: case-insensitive, allow colon spacing variants
    pattern = re.compile(re.escape(tgt_label) + r"\s*:\s*", re.IGNORECASE)
    m = pattern.search(text)
    if m:
        text = text[m.end():].strip()
    text = text.replace("\n", " ").strip()
    text = re.sub(r"^\s*[-`]+\s*", "", text)
    return text
'''



# ---------------------------------------------------------------------------
# 7a.  STANDARD HuggingFace MODEL LOADING & INFERENCE
# ---------------------------------------------------------------------------

def load_model_hf(model_path: str):
    print(f"[HF] Loading model from {model_path} ...")
    hf_tok = AutoTokenizer.from_pretrained(model_path)

    if hf_tok.pad_token is None:
        hf_tok.pad_token = hf_tok.eos_token
    hf_tok.padding_side = "left"

    model  = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, hf_tok, None   # (model, tokenizer, patcher=None)


@torch.no_grad()
def run_inference_hf(
    model, hf_tok, prompts: list[str],
    max_new_tokens: int = 256, batch_size: int = 8,
) -> list[str]:
    """Standard HF batched greedy decode; returns only the generated tokens."""
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = hf_tok(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(model.device)
        input_len = enc["input_ids"].shape[1]
        gen_ids   = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=hf_tok.eos_token_id,
        )
        decoded = hf_tok.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)
        outputs.extend(decoded)
        if (i // batch_size) % 10 == 0:
            print(f"  [HF] {i + len(batch)}/{len(prompts)} done")
    return outputs


# ---------------------------------------------------------------------------
# 7b.  BLT MODEL LOADING & INFERENCE  (bytelatent library)
# ---------------------------------------------------------------------------

def load_model_blt(model_path: str, entropy_model_path: str):
    """
    Load a BLT checkpoint using the bytelatent library.

    Returns (model, blt_tokenizer, patcher).

    Mirrors the reference script exactly:
      1.  Set up torch.distributed (required by bytelatent even for single-GPU)
      2.  load_consolidated_model_and_tokenizer
      3.  Build the patcher with realtime_patching=True
    """
    torch._dynamo.config.suppress_errors = True
    distributed_args = DistributedArgs()
    distributed_args.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(distributed_args)

    print(f"[BLT] Loading model from {model_path} ...")
    model, blt_tok, train_cfg = load_consolidated_model_and_tokenizer(model_path)

    assert isinstance(model, ByteLatentTransformer), (
        f"Expected ByteLatentTransformer, got {type(model)}"
    )
    assert isinstance(blt_tok, BltTokenizer), (
        f"Expected BltTokenizer, got {type(blt_tok)}"
    )

    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    patcher_args.entropy_model_checkpoint_dir = entropy_model_path
    print(f"[BLT] Building patcher with entropy model from {entropy_model_path} ...")
    patcher = patcher_args.build()

    return model, blt_tok, patcher


@torch.no_grad()
def run_inference_blt(
    model, blt_tok, patcher, prompts: list[str],
    max_new_tokens: int = 256, batch_size: int = 8,
) -> list[str]:
    """
    BLT inference via generate_nocache.

    generate_nocache returns full byte sequences (prompt + completion).
    We decode, then strip the prompt prefix to obtain only the completion,
    matching what run_inference_hf returns.

    Notes:
      - max_gen_len is passed as a keyword arg; if generate_nocache does not
        accept it, remove the kwarg and control length via model config instead.
      - BLT operates on raw bytes so no HF tokenizer padding/truncation is needed.
    """
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]

        batch = [p if p.strip() else " " for p in batch]

        out_tokens = generate_nocache(
            batch,
            model=model,
            tokenizer=blt_tok,
            patcher=patcher,
            max_gen_len=max_new_tokens,
        )

        for tokens in out_tokens:
            outputs.append(blt_tok.decode(tokens))

        if (i // batch_size) % 10 == 0:
            print(f"  [BLT] {i + len(batch)}/{len(prompts)} done")

    return outputs


# ---------------------------------------------------------------------------
# 8.  UNIFIED LOAD + INFERENCE  (dispatches on tokenizer_name)
# ---------------------------------------------------------------------------

def load_model(tokenizer_name: str, model_path: str, entropy_model_path: str):
    if tokenizer_name == "blt":
        return load_model_blt(model_path, entropy_model_path)
    else:
        return load_model_hf(model_path)


def run_inference(
    tokenizer_name: str, model, tokenizer, patcher,
    prompts: list[str], max_new_tokens: int, batch_size: int,
) -> list[str]:
    if tokenizer_name == "blt":
        return run_inference_blt(model, tokenizer, patcher, prompts, max_new_tokens, batch_size)
    else:
        return run_inference_hf(model, tokenizer, prompts, max_new_tokens, batch_size)


# ---------------------------------------------------------------------------
# 9.  SINGLE DIRECTION EVALUATION
# ---------------------------------------------------------------------------

def evaluate_direction(
    tokenizer_name: str, model, tokenizer, patcher,
    src_lang: str, tgt_lang: str,
    flores_dir: str, max_new_tokens: int, batch_size: int,
) -> dict:
    print(f"\n=== {src_lang} -> {tgt_lang} ===")

    src_sents, tgt_sents_raw = load_flores_local(src_lang, tgt_lang, flores_dir)
    print(f"  {len(src_sents)} test sentences")

    prompts   = [build_prompt(src_lang, tgt_lang, s) for s in src_sents]
    raw_out   = run_inference(tokenizer_name, model, tokenizer, patcher, prompts, max_new_tokens, batch_size)


    tgt_label = LANG_PAIR_META[(src_lang, tgt_lang)]["tgt_label"]
    hyps_raw  = [extract_translation(o, tgt_label) for o in raw_out]

    # spBLEU and chrF++ always on raw strings
    sp   = compute_spbleu(hyps_raw, tgt_sents_raw)
    chrf = compute_chrf(hyps_raw, tgt_sents_raw)

    # BLEU: pretokenize for script-based target languages
    if tgt_lang in PRETOK_LANGS:
        print(f"  Pre-tokenizing for BLEU ({tgt_lang}) ...")
        hyps_tok = pretokenize_corpus(hyps_raw, tgt_lang)
        refs_tok = pretokenize_corpus(tgt_sents_raw, tgt_lang)
    else:
        hyps_tok = hyps_raw
        refs_tok = tgt_sents_raw

    bleu = compute_bleu(hyps_tok, refs_tok, tgt_lang)

    print(f"  BLEU {bleu:.4f}  |  spBLEU {sp:.4f}  |  chrF++ {chrf:.4f}")
    return {
        "src": src_lang, "tgt": tgt_lang,
        "direction": f"{src_lang}->{tgt_lang}",
        "bleu":   round(bleu, 4),
        "spbleu": round(sp,   4),
        "chrf":   round(chrf, 4),
        "hypotheses": hyps_raw,
        "references": tgt_sents_raw,
    }


# ---------------------------------------------------------------------------
# 10. FULL EVALUATION LOOP
# ---------------------------------------------------------------------------

def run_full_evaluation(
    tokenizer_name: str,
    model_path: str,
    entropy_model_path: str,
    flores_dir: str,
    output_dir: str,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    directions: Optional[list[tuple]] = None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model, tokenizer, patcher = load_model(tokenizer_name, model_path, entropy_model_path)

    if directions is None:
        directions = [("en", tgt) for tgt in SEA_LANGS] + [(src, "en") for src in SEA_LANGS]


    results = []
    for src_lang, tgt_lang in directions:
        res = evaluate_direction(
            tokenizer_name, model, tokenizer, patcher,
            src_lang, tgt_lang, flores_dir, max_new_tokens, batch_size,
        )
        Path(output_dir).joinpath(f"{tokenizer_name}_{src_lang}-{tgt_lang}_hyps.txt").write_text(
            "\n".join(res["hypotheses"]), encoding="utf-8"
        )
        results.append({k: v for k, v in res.items() if k not in ("hypotheses", "references")})
        results[-1]["tokenizer"] = tokenizer_name

    en_xx = [r for r in results if r["src"] == "en"]
    xx_en = [r for r in results if r["tgt"] == "en"]

    def avg(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None

    summary = {
        "tokenizer":          tokenizer_name,
        "en_xx_bleu_avg":     avg(en_xx, "bleu"),
        "en_xx_spbleu_avg":   avg(en_xx, "spbleu"),
        "en_xx_chrf_avg":     avg(en_xx, "chrf"),
        "xx_en_bleu_avg":     avg(xx_en, "bleu"),
        "xx_en_spbleu_avg":   avg(xx_en, "spbleu"),
        "xx_en_chrf_avg":     avg(xx_en, "chrf"),
    }

    csv_path = Path(output_dir) / f"{tokenizer_name}_results.csv"
    fieldnames = ["tokenizer", "direction", "src", "tgt", "bleu", "spbleu", "chrf"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in fieldnames} for r in results])

    Path(output_dir).joinpath(f"{tokenizer_name}_summary.json").write_text(
        json.dumps({**summary, "per_direction": results}, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print(f"RESULTS — {tokenizer_name}")
    print(f"  {'Direction':<12} {'BLEU':>7} {'spBLEU':>8} {'chrF++':>8}")
    print("  " + "-" * 40)
    for r in results:
        print(f"  {r['direction']:<12} {r['bleu']:>7.4f} {r['spbleu']:>8.4f} {r['chrf']:>8.4f}")
    print("  " + "-" * 40)
    print(f"  EN->XX avg   {summary['en_xx_bleu_avg']:>7.4f} {summary['en_xx_spbleu_avg']:>8.4f} {summary['en_xx_chrf_avg']:>8.4f}")
    print(f"  XX->EN avg   {summary['xx_en_bleu_avg']:>7.4f} {summary['xx_en_spbleu_avg']:>8.4f} {summary['xx_en_chrf_avg']:>8.4f}")
    print(f"\nSaved to: {output_dir}")
    return results, summary


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="FLORES+ translation eval — BLEU + spBLEU + chrF++ (OpenSEAL-compatible)"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to model checkpoint (HF or BLT consolidated)")
    parser.add_argument("--tokenizer_name", required=True,
                        choices=["pabpe", "myte", "blt", "blbpe"])
    parser.add_argument(
        "--entropy_model_path",
        default=None,
        help="[BLT only] Path to the entropy model directory (e.g. hf-weights/entropy_model). "
             "Required when --tokenizer_name blt.",
    )
    parser.add_argument(
        "--flores_dir",
        default="/scratch/Projects/CFP-01/CFP01-CF-060/kieron/data/flores-plus_dev_devtest",
        help="Local FLORES+ folder containing *.devtest files",
    )
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--langs", nargs="+", default=None,
                        help="Subset of SEA lang codes (default: all 10)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.tokenizer_name == "blt" and args.entropy_model_path is None:
        raise ValueError("--entropy_model_path is required when --tokenizer_name is blt")

    directions = None
    if args.langs:
        directions = (
            [("en", tgt) for tgt in args.langs] +
            [(src, "en") for src in args.langs]
        )

    run_full_evaluation(
        tokenizer_name=args.tokenizer_name,
        model_path=args.model_path,
        entropy_model_path=args.entropy_model_path or "",
        flores_dir=args.flores_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        directions=directions,
    )