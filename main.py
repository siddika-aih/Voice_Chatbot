from input_data import speech_to_text
from data_store import data_extract_chunk,data_store_vectorstore
from generate_rag_response import generate_rag_response
from output_data import text_to_speech
def main():
    print("Hello from voicebot!")



    speech_to_text()
    result=generate_rag_response(speech_to_text)
    text_to_speech(result)


if __name__ == "__main__":
    main()
