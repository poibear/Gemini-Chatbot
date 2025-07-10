# Gemini Chatbot

## A simple, real-time dialogue application between you and Google's Gemini LLM with audio input.

This is a simple Python implementation of using Gemini as a chatbot by recording your prompts as audio, converting them to text, and receiving the AI's response through a TTS model (via Coqui.ai). 

<details><summary>How is speech transcribed into text to prompt Gemini?</summary>
Funny enough (with some discovery), Gemini will automatically transcribe any uploaded audio files to the best of its abilities without additional prompting. By uploading an audio file of your prompt, Gemini will return a response to what you said. Albeit it is not perfect, it easily gets the job done and does not require any extra libraries compared to other methods.
</details>

### How does it work?
0. The program runs in your console, so no GUI is involved.
1. You select your desired audio input device
2. The program counts down to get you ready to record your prompt
3. The program waits until a certain number of seconds of silence has passed while recording
4. It saves your recording to a file and uploads it to Gemini to transcribe your words
5. The program feeds the "response" from Gemini (the transcribed prompt) to Gemini again to get Gemini's actual reply to your prompt
6. A TTS model reads out Gemini's message, with or without voice conversion
   1. Voice conversion requires inference using PyTorch, **your GPU** will be used if applicable, and with the right drivers/tools for the best performance
7. You record your message again to respond, and the process repeats at #3

### Usage
To run the program, simply run `ai_tts.py`.

(All instructions for using the program are in the base class `AICompanion`)

### Requirements
Packages Used:
```
- coqui-tts
- PyAudio
- google-generativeai
- pathlib
- python-dotenv
```
If you prefer installing from a `requirements.txt` file, there is an attached file. 

Simply run:

`pip install -r requirements.txt`

in the working directory to get all the packages (best for CUDA installations).

(If you have issues installing PyAudio, especially as a Linux user, please see this [announcement](https://github.com/poibear/Gemini-Chatbot/discussions/1) for a fix.)
