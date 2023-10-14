from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import requests
from google.colab import files

def textModel(TEXT,source,target):
  # Load the model and tokenizer
  model_name = "facebook/nllb-200-distilled-600M"
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  translator = pipeline('translation', model=model, tokenizer=tokenizer, max_length=400)

  language_codes = {
    "acehnese (Arabic script)": "ace_Arab",
    "acehnese (Latin script)": "ace_Latn",
    "mesopotamian Arabic": "acm_Arab",
    "ta’izzi-Adeni Arabic": "acq_Arab",
    "tunisian Arabic": "aeb_Arab",
    "afrikaans": "afr_Latn",
    "south Levantine Arabic": "ajp_Arab",
    "akan": "aka_Latn",
    "amharic": "amh_Ethi",
    "north Levantine Arabic": "apc_Arab",
    "modern Standard Arabic": "arb_Arab",
    "modern Standard Arabic (Romanized)": "arb_Latn",
    "najdi Arabic": "ars_Arab",
    "moroccan Arabic": "ary_Arab",
    "egyptian Arabic": "arz_Arab",
    "assamese": "asm_Beng",
    "asturian": "ast_Latn",
    "awadhi": "awa_Deva",
    "central Aymara": "ayr_Latn",
    "south Azerbaijani": "azb_Arab",
    "north Azerbaijani": "azj_Latn",
    "bashkir": "bak_Cyrl",
    "bambara": "bam_Latn",
    "balinese": "ban_Latn",
    "belarusian": "bel_Cyrl",
    "bemba": "bem_Latn",
    "bengali": "ben_Beng",
    "bhojpuri": "bho_Deva",
    "banjar (Arabic script)": "bjn_Arab",
    "banjar (Latin script)": "bjn_Latn",
    "standard Tibetan": "bod_Tibt",
    "bosnian": "bos_Latn",
    "buginese": "bug_Latn",
    "bulgarian": "bul_Cyrl",
    "catalan": "cat_Latn",
    "cebuano": "ceb_Latn",
    "czech": "ces_Latn",
    "chokwe": "cjk_Latn",
    "central Kurdish": "ckb_Arab",
    "crimean Tatar": "crh_Latn",
    "welsh": "cym_Latn",
    "danish": "dan_Latn",
    "german": "deu_Latn",
    "southwestern Dinka": "dik_Latn",
    "dyula": "dyu_Latn",
    "dzongkha": "dzo_Tibt",
    "greek": "ell_Grek",
    "english": "eng_Latn",
    "esperanto": "epo_Latn",
    "estonian": "est_Latn",
    "basque": "eus_Latn",
    "ewe": "ewe_Latn",
    "faroese": "fao_Latn",
    "fijian": "fij_Latn",
    "finnish": "fin_Latn",
    "fon": "fon_Latn",
    "french": "fra_Latn",
    "friulian": "fur_Latn",
    "nigerian Fulfulde": "fuv_Latn",
    "scottish Gaelic": "gla_Latn",
    "irish": "gle_Latn",
    "galician": "glg_Latn",
    "guarani": "grn_Latn",
    "gujarati": "guj_Gujr",
    "haitian Creole": "hat_Latn",
    "hausa": "hau_Latn",
    "hebrew": "heb_Hebr",
    "hindi": "hin_Deva",
    "chhattisgarhi": "hne_Deva",
    "croatian": "hrv_Latn",
    "hungarian": "hun_Latn",
    "armenian": "hye_Armn",
    "igbo": "ibo_Latn",
    "ilocano": "ilo_Latn",
    "indonesian": "ind_Latn",
    "icelandic": "isl_Latn",
    "italian": "ita_Latn",
    "javanese": "jav_Latn",
    "japanese": "jpn_Jpan",
    "kabyle": "kab_Latn",
    "jingpho": "kac_Latn",
    "kamba": "kam_Latn",
    "kannada": "kan_Knda",
    "kashmiri (Arabic script)": "kas_Arab",
    "kashmiri (Devanagari script)": "kas_Deva",
    "georgian": "kat_Geor",
    "central Kanuri (Arabic script)": "knc_Arab",
    "central Kanuri (Latin script)": "knc_Latn",
    "kazakh": "kaz_Cyrl",
    "kabiyè": "kbp_Latn",
    "kabuverdianu": "kea_Latn",
    "khmer": "khm_Khmr",
    "kikuyu": "kik_Latn",
    "kinyarwanda": "kin_Latn",
    "kyrgyz": "kir_Cyrl",
    "kimbundu": "kmb_Latn",
    "northern Kurdish": "kmr_Latn",
    "kikongo": "kon_Latn",
    "korean": "kor_Hang",
    "lao": "lao_Laoo",
    "ligurian": "lij_Latn",
    "limburgish": "lim_Latn",
    "lingala": "lin_Latn",
    "lithuanian": "lit_Latn",
    "lombard": "lmo_Latn",
    "latgalian": "ltg_Latn",
    "luxembourgish": "ltz_Latn",
    "luba-Kasai": "lua_Latn",
    "ganda": "lug_Latn",
    "luo": "luo_Latn",
    "mizo": "lus_Latn",
    "standard Latvian": "lvs_Latn",
    "magahi": "mag_Deva",
    "maithili": "mai_Deva",
    "malayalam": "mal_Mlym",
    "marathi": "mar_Deva",
    "minangkabau (Arabic script)": "min_Arab",
    "minangkabau (Latin script)": "min_Latn",
    "macedonian": "mkd_Cyrl",
    "plateau Malagasy": "plt_Latn",
    "maltese": "mlt_Latn",
    "meitei (Bengali script)": "mni_Beng",
    "halh Mongolian": "khk_Cyrl",
    "mossi": "mos_Latn",
    "maori": "mri_Latn",
    "burmese": "mya_Mymr",
    "dutch": "nld_Latn",
    "norwegian Nynorsk": "nno_Latn",
    "norwegian Bokmål": "nob_Latn",
    "nepali": "npi_Deva",
    "northern Sotho": "nso_Latn",
    "nuer": "nus_Latn",
    "nyanja": "nya_Latn",
    "occitan": "oci_Latn",
    "west Central Oromo": "gaz_Latn",
    "odia": "ory_Orya",
    "pangasinan": "pag_Latn",
    "eastern Panjabi": "pan_Guru",
    "papiamento": "pap_Latn",
    "western Persian": "pes_Arab",
    "polish": "pol_Latn",
    "portuguese": "por_Latn",
    "dari": "prs_Arab",
    "southern Pashto": "pbt_Arab",
    "ayacucho Quechua": "quy_Latn",
    "romanian": "ron_Latn",
    "rundi": "run_Latn",
    "russian": "rus_Cyrl",
    "sango": "sag_Latn",
    "sanskrit": "san_Deva",
    "santali": "sat_Olck",
    "sicilian": "scn_Latn",
    "shan": "shn_Mymr",
    "sinhala": "sin_Sinh",
    "slovak": "slk_Latn",
    "slovenian": "slv_Latn",
    "samoan": "smo_Latn",
    "shona": "sna_Latn",
    "sindhi": "snd_Arab",
    "somali": "som_Latn",
    "southern Sotho": "sot_Latn",
    "spanish": "spa_Latn",
    "tosk Albanian": "als_Latn",
    "sardinian": "srd_Latn",
    "serbian": "srp_Cyrl",
    "swati": "ssw_Latn",
    "sundanese": "sun_Latn",
    "swedish": "swe_Latn",
    "swahili": "swh_Latn",
    "silesian": "szl_Latn",
    "tamil": "tam_Taml",
    "tatar": "tat_Cyrl",
    "telugu": "tel_Telu",
    "tajik": "tgk_Cyrl",
    "tagalog": "tgl_Latn",
    "thai": "tha_Thai",
    "tigrinya": "tir_Ethi",
    "tamasheq (Latin script)": "taq_Latn",
    "tamasheq (Tifinagh script)": "taq_Tfng",
    "tok Pisin": "tpi_Latn",
    "tswana": "tsn_Latn",
    "tsonga": "tso_Latn",
    "turkmen": "tuk_Latn",
    "tumbuka": "tum_Latn",
    "turkish": "tur_Latn",
    "twi": "twi_Latn",
    "central Atlas Tamazight": "tzm_Tfng",
    "uyghur": "uig_Arab",
    "ukrainian": "ukr_Cyrl",
    "umbundu": "umb_Latn",
    "urdu": "urd_Arab",
    "northern Uzbek": "uzn_Latn",
    "venetian": "vec_Latn",
    "vietnamese": "vie_Latn",
    "waray": "war_Latn",
    "wolof": "wol_Latn",
    "xhosa": "xho_Latn",
    "eastern Yiddish": "ydd_Hebr",
    "yoruba": "yor_Latn",
    "yue Chinese": "yue_Hant",
    "chinese (Simplified)": "zho_Hans",
    "chinese (Traditional)": "zho_Hant",
    "standard Malay": "zsm_Latn",
    "zulu": "zul_Latn"
  }

  input_text = TEXT
  src_lang = source
  tgt_lang = target
  src_lang_code = language_codes.get(src_lang)
  tgt_lang_code = language_codes.get(tgt_lang)

  if src_lang_code is None or tgt_lang_code is None:
      print("Invalid source or target language.")
  else:
      translated_text = translator(input_text, src_lang=src_lang_code, tgt_lang=tgt_lang_code)[0]["translation_text"]
      print("Translated Text:")
      print(translated_text)
option = input('Enter 1 for transtale text to text enter 2 for translate audio to text : ')

if option == '1':
  TEXT = input("Enter the text you want to translate: ").lower()
  src = input("Enter the source language: ").lower()
  tgt = input("Enter the target language: ").lower()
  textModel(TEXT,src,tgt)
elif option == '2':
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
    headers = {"Authorization": "Bearer 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"}  # Replace YOUR_API_KEY with your actual Hugging Face API key

    def query(filename):
        with open(filename, "rb") as f:
            files = {'file': (filename, f)}
            response = requests.post(API_URL, headers=headers, files=files)
            return response.json()

    uploaded = files.upload()
    if len(uploaded) == 0:
        print("No file uploaded")
    else:
        # Assuming only one file is uploaded, you can loop through uploaded.values() to handle multiple files
        filename = list(uploaded.keys())[0]
        output = query(filename)
        output= output['text']
        tgt = input("Enter the target language: ").lower()
    textModel(output,'english',tgt)
else:
  print("select valid option")
