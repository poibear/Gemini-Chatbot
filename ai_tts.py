import os
import re
from TTS.api import TTS  # using fork https://github.com/idiap/coqui-ai-TTS
import sys
import wave
import time
import torch
import pyaudio
import google.generativeai as genai
from struct import pack
from array import array
from pathlib import Path
from sys import byteorder
from dotenv import load_dotenv

"""Note: According to Google, Gemini cannot be prompted by audio files alone. The cheesiest workaround that works everytime is using Gemini as a Speech-To-Text
Translator (without additional instructions) and using the text it outputs as our prompt to Gemini. This requires two uses of the generate_content method..."""

class AICompanion:
    
    gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]  # if outdated, you can check https://ai.google.dev/gemini-api/docs/models/gemini #model-variations
    
    def __init__(self,
                 ai_model: str = "gemini-1.5-pro-latest",
                 base_tts: str = "tts_models/multilingual/multi-dataset/xtts_v2",
                 prompt_filepath: str | os.PathLike = Path.joinpath(Path.cwd(), "recordings", "active_prompt.wav"), 
                 output_filepath: str | os.PathLike = Path.joinpath(Path.cwd(), "recordings", "voice_output.wav"),
                 buffer_size: int = 4096,
                 env_file_loc: str | os.PathLike = None,
                 device: str="cuda" if torch.cuda.is_available() else "cpu",
                 voice_conversion: bool = False,
                 speaker_language: str = "en",
                 desired_speaker_wav: str | os.PathLike = None
                 ):
        
        """A realtime AI companion powered by Google's Gemini LLM to converses in dialogue, with optional voice conversion powered by Coqui.AI's TTS.
        NOTE: You need to define your API key for Gemini as an environment variable called GEMINI_APIKEY,
        available through your console or defining the filepath of your .env file in the parameter env_file_loc.
        For voice conversion, set the variable "voice_conversion" to True, and set parameters following it accordingly.
        
        Args:
            ai_model (str, optional): A modelname supported by Google to define what type of Gemini model to use
            base_tts (str, optional): The base TTS model to use from Coqui.ai's AI models (use tts --list_models to view available models)
            prompt_filepath (str | os.PathLike, optional): The filepath that determines where to place recordings of the user's audio prompts with the AI
            output_filepath (str | os.PathLike, optional): The filepath that determines where to place recordings of the AI's audio responses
            buffer_size (int, optional): Determines the buffer size for PyAudio to process WAV streams (recording and playing)
            env_file_loc (str | None, optional): Defines the path to the env file storing a Google API key to use Gemini
            device (str, optional): Whether to utilize your GPU for voice inference (voice changer)
            voice_conversion (bool, optional): Whether to use voice conversion when reading prompts from the AI
            speaker_language (str, optional): The locale for the TTS model to speak in (if supported by the TTS model)
            desired_speaker_wav (str | os.PathLike, optional): The path (relative/absolute) to the WAV file of someone speaking for the TTS model to clone
        """
        # load gemini apikey from .env file if applicable
        self.env_file_loc = env_file_loc
        load_dotenv(self.env_file_loc)
        
        # register api key for google's gemini
        genai.configure(api_key=os.getenv("GEMINI_APIKEY"))
        
        # add gemini instance
        self.gclient = genai.GenerativeModel(ai_model)  # might need pro to allow audio prompting if not working
        
        # no ai object when api key isnt defined, usually returns error but we will catch it anyways
        if self.gclient is None:
            raise KeyError("There is no Gemini API key. Provide your Gemini API key through the environment variable \"GEMINI_APIKEY\" in your console or through a .env file.")
        
        # add location for where to store audio file prompts & results
        self.prompt_fileloc = prompt_filepath
        self.output_fileloc = output_filepath
        
        # configure pyaudio for later
        self.input_device = self._choose_input_device()
        self.buffer_size = buffer_size
        
        self.base_tts = base_tts  # for info reasons
        self.model_tts = TTS(self.base_tts).to(device)
        
        self.desired_speaker = desired_speaker_wav
        self.speaker_language = speaker_language
        self.voice_conversion = voice_conversion


    @property
    def info(self) -> dict:
        """Outputs brief information about the AI companion (e.g., Gemini Model type) instance as a dictionary."""
        information = {
            "ai_variant": self.gclient.model_name,
            "tts_model": self.base_tts,
            "input_device": self.input_device,
            "env_file_loc": self.env_file_loc  # can be None, otherwise a Path obj
        }
        
        return information


    def _choose_input_device(self) -> int:
        """Return the desired input device to use for the AI conversation session as an index number in compliance with PyAudio's available selection."""
        # get input device count to iterate through
        pA = pyaudio.PyAudio()
        device_count: int = pA.get_host_api_info_by_index(0).get('deviceCount')  # type: ignore  # how can i have a float device count, do i have half an input device or .0 devices??
        
        print("Input Device Index - Input Device Name")
        for device_index in range(0, device_count):
            device = pA.get_device_info_by_host_api_device_index(0, device_index)
            if (device.get("maxInputChannels")) > 0:  # if device in input device dict isn't emulated input
                print(f"{device_index} - {device.get('name', 'No Name Device')}")
        
        while True:
            try: 
                device_selection = int(input("Type in the input device's index you wish to use for this session: "))
                break
        
            except ValueError:
                print("Enter a valid number")
                continue
        
        return device_selection


    # next 3 methods are from stackoverflow
    # From StackOverflow - https://stackoverflow.com/questions/892199/detect-record-audio-in-python/ - Asked by Sam Machin - Answered by cryo - https://stackoverflow.com/users/304185/cryo
    def _check_input_silence(self, audio_data: list, threshold: int=1000) -> bool:
        """Determines whether the audio data is below the threshold
        
        Args:
            audio_data (list): The array of audio data from the input device to analyze
            threshold (int, optional): The integer acting as the lowest limit for the audio data to be above
        """
        return max(audio_data) < threshold


    def record_raw_audio(self,
                         audio_format: int,
                         audio_channels: int,
                         audio_bitrate: int,
                         silence_allowed: int) -> tuple[int, array]:
        """Records the raw audio from an input device and appends it to a (Python) list. Returns sample width of the recording and the list containing the recording's audio data."""
        recording: array = array('h')  # signed (pos & negative) short integer array
        recording_started = False
        seconds_silent = 0
        
        pA = pyaudio.PyAudio()
        
        audio_stream = pA.open(
            format=audio_format,
            channels=audio_channels,
            rate=audio_bitrate,
            input=True,
            frames_per_buffer=self.buffer_size,
            input_device_index=self.input_device
        )
        
        while True:
            activity = array('h', audio_stream.read(self.buffer_size))
            # endianness, usually determined by if cpu arch is mac/linux or windows-like
            if byteorder == "big":
                activity.byteswap()
            recording.extend(activity)
            
            is_silent = self._check_input_silence(activity)
            
            if not is_silent and not recording_started:  # if the recording initially captures noise and we didnt know already, base case and only happens once
                recording_started = True
                
            elif is_silent and recording_started:  # if the user suddenly goes quiet while the recording is in progress
                seconds_silent += 1
                
            if recording_started and seconds_silent > silence_allowed:  # if during the recording the user is quiet for a certain amount of time (e.g., 3 secs)
                break
        
        # stop everything
        audio_stream.close()
        
        # include sample width for exporting
        sample_width = pA.get_sample_size(audio_format)
        
        pA.terminate()
        
        return (sample_width, recording)


    def record_to_file(self,
                       audio_format=pyaudio.paInt32,
                       audio_channels: int=2,
                       audio_bitrate: int=44100,
                       silence_allowed: int=1):
        """
        Record an audio prompt from the user, automatically ending the recording with a sufficient silence duration, to send to Google's Gemini through a WAV file.
        Args:
            audio_format (int, optional): Defines the audio format (in  # of bits per sample) to use when recording the user's prompt
            audio_channels (int, optional): Defines the number of audio channels to use when recording
            audio_bitrate (int, optional): The bitrate to use when recording
            silence_allowed (int, optional): The number of milliseconds of silence to allow before ending the recording"""
        # silence_allowed *= 10
        if Path.exists(self.prompt_fileloc):  # just in case to prevent file issues
            Path.unlink(self.prompt_fileloc)  # delete file
            
        sample_width, raw_audio = self.record_raw_audio(audio_format,
                                                        audio_channels,
                                                        audio_bitrate,
                                                        silence_allowed)
        raw_audio = pack('<' + ('h' * len(raw_audio)), *raw_audio)  # audio array to binary for wave to write to
        
        # setup & configuration
        prompt_wav: wave.Wave_write = wave.open(str(self.prompt_fileloc), 'wb')  # cast string since wave cant handle Path
        prompt_wav.setnchannels(audio_channels)
        prompt_wav.setsampwidth(sample_width)
        prompt_wav.setframerate(audio_bitrate)
        
        # actual audio file writing
        prompt_wav.writeframes(raw_audio)
        prompt_wav.close()
        
        return True


    def prompt(self) -> tuple[genai.types.GenerateContentResponse, genai.types.GenerateContentResponse] | genai.protos.GenerateContentResponse.PromptFeedback.BlockReason: 
        """Provide an audio prompt to Google's Gemini AI and play a wav file of the AI's response through the desired RVC model's voice.
        This function will persistently ensure that the audio attachment is uploaded to Gemini.
        Returns your prompt as text and Gemini's response or a reported error if a prompt violates the ToS.
        Note: To get only the response from Gemini (not the entire ResponseClass), simply get the property "text" from the return value (if applicable)
        
        Return Values:
            stt (GenerateContentResponse): Your transcribed words from the prompt wave file processed by Gemini
            ai_response (GenerateContentResponse): Gemini's response to your prompt (as if you used text to talk to it!)
        """
        if not Path.exists(self.prompt_fileloc):
            raise FileNotFoundError(f"Cannot prompt Gemini. There is no recorded audio prompt file in the directory \"{self.prompt_fileloc}\"")
        
        while True:
            prompt_file = genai.upload_file(path=self.prompt_fileloc)
            
            while prompt_file.state.name == "PROCESSING":
                print("Verifying recording has been sent...")
                prompt_file = genai.get_file(prompt_file.name)
            
            if prompt_file.state.name == "FAILED":
                print("Upload failed, retrying...")
                continue
            
            else:  # completed upload
                break

        try:
            stt = self.gclient.generate_content(prompt_file)  # this converts our wav into text lol
            ai_response = self.gclient.generate_content(stt.text)  # this actually provides the feedback
            return (stt, ai_response)
        
        except genai.types.BlockedPromptException:
            return stt.prompt_feedback

      
    def play_prompt(self, response: genai.types.GenerateContentResponse):  # plays wav file
        """Converts text into an audio file with a voice from a given WAV file."""
        # From StackOverflow - https://stackoverflow.com/questions/17657103/play-wav-file-in-python/ - Asked by nim4n - https://stackoverflow.com/users/1836709/nim4n - Answered by zhangyangyu - https://stackoverflow.com/users/2189957/zhangyangyu
        formatted_response = re.sub(r'[^A-Za-z0-9 ]+', '', response.text)  # only allow alphanumeric characters for voice changer
        if self.voice_conversion is True:
            self.model_tts.tts_with_vc_to_file(formatted_response,
                                               language = self.speaker_language,
                                               speaker_wav = self.desired_speaker,
                                               file_path = str(self.output_fileloc))
        elif self.voice_conversion is False:
            self.model_tts.tts_to_file(formatted_response,
                                       file_path = str(self.output_fileloc),
                                       language = self.speaker_language,
                                       speaker_wav = self.desired_speaker)  # return value is implied by file_path param
        
        new_wav: wave.Wave_read = wave.open(str(self.output_fileloc), 'rb')  # read in binary, filepath is initially Path obj, not str
        
        pA = pyaudio.PyAudio()
        
        output_stream = pA.open(format = pA.get_format_from_width(new_wav.getsampwidth()),
                                channels = new_wav.getnchannels(),
                                rate = new_wav.getframerate(),
                                output = True)
         # get audio data
        audio_data = new_wav.readframes(self.buffer_size)
        
         # play everything, uninterrupted
         # i could use walrus operator but this seems more intuitive
        while audio_data:
            output_stream.write(audio_data)
            audio_data = new_wav.readframes(self.buffer_size)
        
         # we done
        output_stream.close()
        
        pA.terminate()


def countdown(seconds: int=3):
    """Create a countdown for the console before performing an action."""
    for number in range(seconds, 0, -1):
        print(number)
        time.sleep(1)

if __name__ == "__main__":
    print("Loading Input Devices...")
    ai_companion = AICompanion(desired_speaker="./models/speechsample.wav",
                               env_file_loc="./gemini_key.env")

    try:
        while True:
            print("Get ready to record your response...")
            countdown()
            print("Record your question/statement to Gemini... (The recording will automatically end when you are silent)")
            ai_companion.record_to_file()
            print("Gemini is responding...", end="\r")
            transcribed_text, response = ai_companion.prompt()
            print(f"You: {transcribed_text.text}", flush=True)
            print(f"Gemini: {response.text}")
            ai_companion.play_prompt(response)
            
    except KeyboardInterrupt:
        sys.exit(0)