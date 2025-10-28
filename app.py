import streamlit as st
from PIL import Image
from ultralytics import YOLO
import base64, io, os
from sambanova import SambaNova
import warnings
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")


try:
    model = YOLO("models/best.pt") 
    sn_client = SambaNova(api_key=os.getenv("API_KEY"), base_url="https://api.sambanova.ai/v1") 
except Exception as e:
    st.error(f"❌ Ошибка инициализации модели или API: {e}. Анализ не будет работать.")
    model = None
    sn_client = None



def get_quick_advice(client, full_advice, topic_question):
    """
    Вызывает LLM для извлечения конкретного совета из полного текста диагноза.
    """
    if not client:
        return f"Не удалось получить совет: SambaNova API недоступен."

    prompt = (
        f"Учитывая следующий полный диагноз и план лечения: «{full_advice}», "
        f"ответьте максимально кратко и точно только на один вопрос: {topic_question}"
    )

    try:
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct",
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }],
            temperature=0.1 
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка при получении совета от ИИ: {e}"



st.set_page_config(page_title="FloraScan AI", page_icon="🌿", layout="centered")
st.title("🌿 FloraScan AI")
st.markdown("Загрузите фото растения, и ИИ определит, есть ли болезнь и подскажет, как лечить.")


if "advice_text" not in st.session_state:
    st.session_state.advice_text = ""
if "image" not in st.session_state:
    st.session_state.image = None
if "image_with_boxes" not in st.session_state:
    st.session_state.image_with_boxes = None
if "detections" not in st.session_state:
    st.session_state.detections = []
if "button_response" not in st.session_state:
    st.session_state.button_response = ""
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""
if "expander_expanded" not in st.session_state:
    st.session_state.expander_expanded = False



uploaded_file = st.file_uploader("📸 **Загрузите изображение растения**", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")
    st.image(st.session_state.image, caption="Ваше фото", use_container_width=True)



if st.session_state.image and st.button("🔍 **Анализировать**", type="primary"):

    st.session_state.button_response = ""
    st.session_state.last_topic = ""
    
    if not model or not sn_client:
        st.error("Анализ невозможен. Проверьте сообщения об ошибках инициализации модели/API выше.")
    else:
        with st.spinner("⏳ Анализ изображения..."):
            results = model(st.session_state.image)
            st.session_state.detections = []
            
            for box in results[0].boxes:
                cls = results[0].names[int(box.cls)]
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                st.session_state.detections.append({
                    "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                    "label": cls, "confidence": conf
                })


            st.session_state.image_with_boxes = Image.fromarray(results[0].plot()[:, :, ::-1])
            st.image(st.session_state.image_with_boxes, caption="🔬 Результаты", use_container_width=True)


            if any(det["label"] == "ill" for det in st.session_state.detections):
                

                st.markdown("---")
                st.subheader("⚠️ Обнаружена болезнь")
                
                buffered = io.BytesIO()
                st.session_state.image.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode()

                try:
                    sn_response = sn_client.chat.completions.create(
                        model="Llama-4-Maverick-17B-128E-Instruct",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Определи болезнь растения и предложи подробный план лечения, включая рекомендации по поливу, освещению и конкретным средствам."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                            ]
                        }],
                        temperature=0.2
                    )
                    st.session_state.advice_text = sn_response.choices[0].message.content
                    
  
                    st.session_state.expander_expanded = True
                        
                except Exception as e:
                    st.error(f"❌ Критическая ошибка при получении рекомендаций от ИИ: {e}")
                    st.session_state.advice_text = ""
            else:
                st.markdown("---")
                st.success("✅ **Отличные новости!** Растение выглядит здоровым. Продолжайте уход.")


if st.session_state.advice_text:
    st.markdown("---")

    with st.expander("💬 **Полный диагноз и план лечения**", expanded=st.session_state.expander_expanded):
        st.markdown(st.session_state.advice_text)

if st.session_state.advice_text:
    st.subheader("⚡ Детализированные советы")
    
    col1, col2, col3 = st.columns(3)

    if col1.button("💧 Как поливать", use_container_width=True):
        st.session_state.last_topic = "Полив"
        st.session_state.button_response = get_quick_advice(sn_client, st.session_state.advice_text, "Как следует поливать растение с этой болезнью?")
        
    
    if col2.button("🌞 Как освещать", use_container_width=True):
        st.session_state.last_topic = "Освещение"
        st.session_state.button_response = get_quick_advice(sn_client, st.session_state.advice_text, "Какой режим освещения оптимален для лечения этой болезни?")
        
    if col3.button("🧴 Какие средства лечния", use_container_width=True):
        st.session_state.last_topic = "Средства лечения"
        st.session_state.button_response = get_quick_advice(sn_client, st.session_state.advice_text, "Назовите конкретный тип или пример препарата, подходящего для лечения этой болезни.")


    if st.session_state.button_response:
        st.markdown(f"**💡 Совет по: {st.session_state.last_topic}**")
        st.success(st.session_state.button_response)