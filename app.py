import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. SAYFA AYARLARI
st.set_page_config(
    page_title="ALKÜ SSS Asistanı",
    page_icon="🎓",
    layout="wide"
)

# 2. VERİ SETİ VE MODEL
faqs = [
    # --- AKADEMİK & SINAVLAR ---
{"question": "Okul ne zaman açılacak?", "answer": "Güz dönemi dersleri genellikle Eylül ayının son haftasında, Bahar dönemi ise Şubat ayının ortasında başlar. Kesin tarihler için ALKÜ web sitesindeki Akademik Takvime bakmalısınız."},
    {"question": "Geçme notu kaçtır?",
     "answer": "ALKÜ'de geçme notu bağıl sisteme (Çan Eğrisi) göre belirlenir, ancak finalden en az 35 alma zorunluluğu vardır."},
    {"question": "Çan eğrisi var mı?",
     "answer": "Evet, mühendislik fakültesi derslerinin çoğunda Çan Eğrisi (Bağıl Değerlendirme) sistemi uygulanır."},
    {"question": "Devamsızlık hakkı ne kadar?",
     "answer": "Teorik derslerde %30, uygulamalı derslerde %20 devamsızlık hakkın vardır. Sınırı aşarsan 'DZ' (Devamsız) notu alırsın ve finale giremezsin."},
    {"question": "Bütünleme sınavı var mı?",
     "answer": "Evet, final sınavında başarısız olanlar veya finale giremeyenler için Bütünleme sınavı yapılır."},
    {"question": "Tek ders sınavı nedir?",
     "answer": "Mezuniyetine tek bir dersi kalan öğrenciler için dönem sonunda yapılan özel sınavdır."},
    {"question": "Ortalama (GNO) kaç olmalı?",
     "answer": "Mezun olabilmek için Genel Not Ortalamasının (GNO) en az 2.00 olması gerekir. 1.80 altı 'Sınamalı' durumdur."},
    {"question": "DC ve DD ile geçilir mi?",
     "answer": "Eğer dönem ortalaman (YNO) 2.00 ve üzerindeyse DC ve DD notu aldığın derslerden geçmiş sayılırsın (Şartlı Geçiş)."},
    {"question": "FF ve FD nedir?",
     "answer": "FF ve FD notları başarısız notlardır. Bu dersleri tekrar alman gerekir."},

    # --- STAJ (ÇOK SORULUR) ---
    {"question": "Staj zorunlu mu?",
     "answer": "Evet, Mühendislik Fakültesi'nde mezuniyet için zorunlu staj (genellikle 40 veya 60 iş günü) vardır."},
    {"question": "Staj defteri ne zaman teslim edilir?",
     "answer": "Staj defterleri, stajın bittiği dönemi takip eden akademik takvimde belirtilen tarihlerde bölüm sekreterliğine teslim edilmelidir."},
    {"question": "Staj yeri buluyor musunuz?",
     "answer": "Staj yerini öğrenci kendisi bulmakla yükümlüdür, ancak kariyer günlerinde firmalarla iletişime geçebilirsiniz."},
    {"question": "Cumartesi staj sayılır mı?",
     "answer": "Eğer iş yerinde Cumartesi günü tam gün mesai yapılıyorsa ve bu belgelenirse staj gününden sayılır."},

    # --- KAMPÜS & YAŞAM ---
    {"question": "Yemekhane rezervasyonu nasıl yapılır?",
     "answer": "Yemekhane rezervasyonları ALKÜ Sağlık Kültür ve Spor Daire Başkanlığı web sitesi üzerinden online olarak yapılır."},
    {"question": "Ring servisleri nereden kalkıyor?",
     "answer": "Kampüs içi ringler ve Alanya merkezden kalkan otobüsler (202, 202-B vb.) fakülteye ulaşım sağlar."},
    {"question": "Kütüphane saatleri nedir?",
     "answer": "Merkez kütüphane genellikle 08.30 - 17.30 arası açıktır, vize ve final haftalarında 7/24 hizmet verebilir."},

    # --- İDARİ İŞLER ---
    {"question": "Öğrenci belgesi nereden alınır?",
     "answer": "Öğrenci belgesini E-Devlet üzerinden veya Öğrenci İşleri Daire Başkanlığı'ndan alabilirsiniz."},
    {"question": "Ders kaydı onaylanmazsa ne olur?",
     "answer": "Danışman onayı yapılmayan ders kayıtları geçersizdir. Ekle-Sil haftasında danışmanınızla iletişime geçmelisiniz."},
    {"question": "ÇAP ve Yandal yapabilir miyim?",
     "answer": "Evet, not ortalaması (GNO) şartlarını (genelde 3.00 ve üzeri) sağlayan öğrenciler Çift Anadal (ÇAP) veya Yandal başvurusu yapabilir."},
    {"question": "Erasmus şartları nelerdir?",
     "answer": "Genel not ortalamasının en az 2.20 olması ve yapılan yabancı dil sınavından yeterli puanın alınması gerekir."},
    {"question": "Harç ücreti var mı?",
     "answer": "Normal öğretim süresi içindeki (4 yıl) devlet üniversitesi öğrencileri harç ödemez. Okulu uzatanlar harç öder."},
]

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)


def get_answer(user_question):
    user_vec = vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    max_similarity = similarity.max()
    index = similarity.argmax()

    if max_similarity < 0.7:
        return "Üzgünüm, bu soru hakkında yeterli bilgim yok. Lütfen fakülte ile ilgili akademik bir soru sorun."
    else:
        return answers[index]


# Oturum Durumu (Hafıza)
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SOL MENÜ (SIDEBAR) ---
with st.sidebar:
    # A. LOGO
    if os.path.exists("Alkulogo.png"):
        st.image("Alkulogo.png", use_container_width=True)
    else:
        st.header("🎓 ALKÜ")

    st.markdown("---")

    # B. YENİ SOHBET BUTONU
    if st.button("🗑️ Sohbeti Temizle", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 🗂️ Geçmiş")

    # C. GİZLİ AÇILIR KAPANIR GEÇMİŞ (EXPANDER)
    # Burası tam istediğin yer: Geçmişi bir kutuya gizler.
    with st.expander("📜 Konuşma Geçmişini Göster"):
        if len(st.session_state.messages) == 0:
            st.write("_Henüz mesaj yok._")
        else:
            # Geçmişi ters sırayla (en yeni en üstte) gösterelim
            for msg in reversed(st.session_state.messages):
                role_icon = "👤" if msg["role"] == "user" else "🤖"
                st.markdown(f"**{role_icon} {msg['role'].title()}:**")
                st.caption(msg["content"])  # Caption, yazıyı biraz daha silik ve küçük yapar
                st.divider()

# --- 4. ANA EKRAN ---
st.title("🎓 ALKÜ MÜHENDİSLİK FAKÜLTESİ Akademik Asistan")
st.write("")

# Burada "st.chat_message" kullanmıyoruz, baloncuk yok!
# Klasik arama kutusu stili:
user_input = st.text_input("Soru", placeholder="Herhangi bir şey sorun ?", label_visibility="collapsed")

if user_input:
    # Cevabı bul
    response = get_answer(user_input)

    # EKRANA YAZDIR (Baloncuksuz, sade kutu içinde)
    if "yeterli bilgim yok" in response:
        st.warning(f"⚠️ **Asistan:** {response}")
    else:
        st.success(f"✅ **Cevap:** {response}")

    # GEÇMİŞE KAYDET (Böylece sol menüde birikir)
    # Aynı soruyu tekrar tekrar kaydetmemesi için son soruyla kontrol edebiliriz
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != response:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.info("💡 **İpucu:** Eski sorularınızı görmek için sol taraftaki menüyü kullanabilirsiniz.")
