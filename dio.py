import base64
import mimetypes
import os
import struct
import pandas as pd
from google import genai
from google.genai import types

def save_binary_file(file_name, data):
    if os.path.exists(file_name):
        print(f"File {file_name} already exists, skipping...")
        return
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
        num_channels, sample_rate, byte_rate,
        block_align, bits_per_sample, b"data", data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    bits_per_sample = 16
    rate = 24000
    
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate = int(param.split("=", 1)[1])
            except:
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except:
                pass
    
    return {"bits_per_sample": bits_per_sample, "rate": rate}

# Variável global para controlar qual API está sendo usada
current_api_index = 0

def generate_audio(text_addition: str, file_index: int):
    global current_api_index
    
    api_keys = ["AlzaSyCIHDcsFPWV-IRn7nJQ0_FPfBa34mw4cs"]
    
    model = "gemini-2.5-flash-preview-tts"
    
    full_prompt = f"""
Voz de um homem 60 anos instrutor, rápido, brasileiro, PT-BR

{text_addition}
"""
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=full_prompt)],
        ),
    ]
    
    config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Orus"
                )
            )
        ),
    )
    
    # Tenta da API atual em diante
    for attempt in range(len(api_keys)):
        api_index = (current_api_index + attempt) % len(api_keys)
        api_key = api_keys[api_index]
        
        try:
            client = genai.Client(api_key=api_key)
            
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    file_name = f"{file_index + 1}"
                    inline_data = part.inline_data
                    data_buffer = inline_data.data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type) or ".wav"
                    if file_extension == ".wav":
                        data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
                    save_binary_file(f"{file_name}{file_extension}", data_buffer)
                else:
                    print(chunk.text)
            
            # Se chegou até aqui, deu sucesso, então retorna
            current_api_index = api_index  # Mantém a API que funcionou
            return
            
        except Exception as e:
            print(f"Erro com a API key {api_keys[api_index][:10]}...: {e}")
            # Se deu erro, marca para tentar a próxima API na próxima chamada
            if attempt == 0:  # Se foi a primeira tentativa (API atual)
                current_api_index = (current_api_index + 1) % len(api_keys)
                print(f"Mudando para próxima API (índice {current_api_index})")
            continue
    
    print("Todas as API keys falharam!")

def main():
    # Lê o arquivo Excel "entradas.xlsx" (coluna única, sem cabeçalho).
    df = pd.read_excel("entradas.xlsx", header=None, dtype=str)
    
    for i, row in df.iterrows():
        # Cada célula da coluna 0 é um parágrafo completo (pode conter quaisquer caracteres)
        text = row[0]
        if isinstance(text, str) and text.strip():
            print(f"Gerando áudio para linha {i}: {text}")
            generate_audio(text.strip(), i)

if __name__ == "__main__":
    main()