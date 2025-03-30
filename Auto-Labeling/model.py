from models.corpus import Corpus
from models.bilstm import BiLSTM
from models.trainer import Trainer
from idsentsegmenter.sentence_segmentation import SentenceSegmentation

# Load Corpus Model
print("Loading Corpus Model...")
corpus = Corpus (
    input_folder = "dataset",
    min_word_freq = 3,
    batch_size = 64,
    wv_file = "models/pretrain/embeddings/wiki.id.case.model"
)

# Load BiLSTM Model
print("Loading BiLSTM Model...")
bilstm = BiLSTM(
    input_dim=len(corpus.word_field.vocab),
    embedding_dim=400,
    char_emb_dim=25,
    char_input_dim=len(corpus.char_field.vocab),
    char_cnn_filter_num=5,
    char_cnn_kernel_size=3,
    hidden_dim=64,
    output_dim=len(corpus.tag_field.vocab),
    lstm_layers=2,
    attn_heads=16,
    emb_dropout=0.5,
    cnn_dropout=0.25,
    lstm_dropout=0.1,
    attn_dropout=0.25,
    fc_dropout=0.25,
    word_pad_idx=corpus.word_pad_idx,
    char_pad_idx=corpus.char_pad_idx,
    tag_pad_idx=corpus.tag_pad_idx
)
bilstm.init_embeddings(
    char_pad_idx=corpus.char_pad_idx,
    word_pad_idx=corpus.word_pad_idx,
    pretrained=corpus.word_field.vocab.vectors if corpus.wv_model else None,
    freeze=True
)
bilstm.init_crf_transitions(
    tag_names=corpus.tag_field.vocab.itos
)

# Load Trainer Model
print("Loading Trainer Model...")
trainer = Trainer(
    model=bilstm,
    data=corpus
)
trainer.load_model("models/model/bilstm_model.pt")

# Input Text
text = "Laporan wartawan Tribunnews.com, Fahdi Fahlevi TRIBUNNEWS.COM, JAKARTA - Ketua Umum Pengurus Besar Persatuan Guru Republik Indonesia (PGRI), Unifah Rosyidi, mengaku setuju atas rencana Pemerintah menerapkan Ujian Nasional (UN). Unifah menilai penerapan UN adalah langkah yang baik sebagai standar penilaian bagi siswa. Meski begitu, Unifah menilai UN bisa diterapkan kembali, tapi tidak menjadi satu-satunya penentu kelulusan. \"Jadi formatnya biar kan para ahli. Tapi itu diperbaiki kaya UN kayak kemarin. Enggak menjadi satu-satunya untuk lulusan. Tetapi menjadi salah satu. Bagaimanapun negara harus hadir dong. Ada standar. Kalau enggak ada standar enggak ada motivasi,\" ujar Unifah kepada wartawan, Senin (2/12/2024). Dirinya mengatakan penerapan kembali adalah upaya memperbaiki sumber daya manusia (SDM) di Indonesia. Menurut Unifah, saat ini terjadi hal yang memalukan saat pelajar Indonesia tidak bisa diterima di tingkat internasional. \"Kan malu kalau sekarang mereka tidak bisa diterima di luar negeri karena kita tidak punya dasar. Kan begitu kan. Jadi bagi kami sih yang utama adalah bagaimana dampaknya bagi masa depan bangsa. Itu yang akan kami bela,\" ucapnya. Namun, Unifah berharap penerapan UN tidak dilakukan kepada siswa Sekolah Dasar (SD). Penerapan UN, menurut Unifah, sebaiknya diterapkan pada jenjang Sekolah Menengah Pertama (SMP) dan Sekolah Menengah Atas (SMA). \"SD itu wajib belajar. Jadi mulailah di SMP. SMP kan untuk ke SMA. SMA untuk ke perguruan tinggi. Jadi seperti itu,\" kata Unifah. Pelaksanaan UN, kata Unifah, bisa dilaksanakan oleh pihak independen. Dirinya menyerahkan pelaksanaan UN dengan formulasi baru kepada Pemerintah. Para siswa, menurut Unifah, akan semangat belajar ketika UN kembali diterapkan. \" Kalau misalnya nilai UN minimum sekian untuk diterima di sini. Itu kan jadi semangat belajar. Begitu juga untuk diintegrasikan dengan perguruan tinggi,\" pungkasnya."

# Sentence segmentation
sentence_segmenter = SentenceSegmentation()
sentences = sentence_segmenter.get_sentences(text)
for sent in enumerate(sentences):
    tokens, predicted_tags, unknown_tokens = trainer.infer(sent[1])