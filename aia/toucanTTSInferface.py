import itertools
import os
import torch
import sys
import soundfile

from InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator
from InferenceInterfaces.InferenceArchitectures.InferenceBigVGAN import BigVGAN
from InferenceInterfaces.InferenceArchitectures.InferenceToucanTTS import ToucanTTS
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id
from TrainingInterfaces.Spectrogram_to_Embedding.StyleEmbedding import StyleEmbedding
MODELS_DIR = "models/toucan"

class ToucanTTSInterface(torch.nn.Module):

    def __init__(self,
                 device="cpu",  # device that everything computes on. If a cuda device is available, this can speed things up by an order of magnitude.
                 tts_model_path=os.path.join(MODELS_DIR, f"ToucanTTS_Meta", "best.pt"),  # path to the ToucanTTS checkpoint or just a shorthand if run standalone
                 embedding_model_path=None,
                 vocoder_model_path=None,  # path to the hifigan/avocodo/bigvgan checkpoint
                 faster_vocoder=True,  # whether to use the quicker HiFiGAN or the better BigVGAN
                 language="en",  # initial language of the model, can be changed later with the setter methods
                 ):
        super().__init__()
        self.device = device
        if not tts_model_path.endswith(".pt"):
            # default to shorthand system
            tts_model_path = os.path.join(MODELS_DIR, f"ToucanTTS_{tts_model_path}", "best.pt")
        if vocoder_model_path is None:
            if faster_vocoder:
                vocoder_model_path = os.path.join(MODELS_DIR, "Avocodo", "best.pt")
            else:
                vocoder_model_path = os.path.join(MODELS_DIR, "BigVGAN", "best.pt")

        ################################
        #   build text to phone        #
        ################################
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True)

        ################################
        #   load weights               #
        ################################
        print('TOUCAN: Loading weights')
        checkpoint = torch.load(tts_model_path, map_location='cpu')

        ################################
        #   load phone to mel model    #
        ################################
        self.use_lang_id = True
        try:
            self.phone2mel = ToucanTTS(weights=checkpoint["model"])  # multi speaker multi language
        except RuntimeError:
            print('TOUCAN: RuntimeError')
            try:
                self.use_lang_id = False
                self.phone2mel = ToucanTTS(weights=checkpoint["model"], lang_embs=None)  # multi speaker single language
            except RuntimeError:
                self.phone2mel = ToucanTTS(weights=checkpoint["model"], lang_embs=None, utt_embed_dim=None)  # single speaker
        with torch.no_grad():
            self.phone2mel.store_inverse_all()  # this also removes weight norm
        print('TOUCAN: load phone2mel')
        self.phone2mel = self.phone2mel.to(torch.device(device))

        #################################
        #  load mel to style models     #
        #################################
        self.style_embedding_function = StyleEmbedding()
        print('TOUCAN: Loading mel to style models')
        if embedding_model_path is None:
            check_dict = torch.load(os.path.join(MODELS_DIR, "Embedding", "embedding_function.pt"), map_location="cpu")
        else:
            check_dict = torch.load(embedding_model_path, map_location="cpu")
        self.style_embedding_function.load_state_dict(check_dict["style_emb_func"])
        self.style_embedding_function.to(self.device)

        ################################
        #  load mel to wave model      #
        ################################
        print('TOUCAN: Loading mel to wave models')
        if faster_vocoder:
            self.mel2wav = HiFiGANGenerator(path_to_weights=vocoder_model_path).to(torch.device(device))
        else:
            self.mel2wav = BigVGAN(path_to_weights=vocoder_model_path).to(torch.device(device))
        self.mel2wav.remove_weight_norm()
        self.mel2wav = torch.jit.trace(self.mel2wav, torch.randn([80, 5]).to(torch.device(device)))

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        self.audio_preprocessor = AudioPreprocessor(input_sr=16000, output_sr=16000, cut_silence=True, device=self.device)
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.style_embedding_function.eval()
        if self.use_lang_id:
            self.lang_id = get_language_id(language)
        else:
            self.lang_id = None
        self.to(torch.device(device))
        self.eval()

    def set_utterance_embedding(self, path_to_reference_audio="", embedding=None):
        print("in set_utterance_embedding")
        if embedding is not None:
            self.default_utterance_embedding = embedding.squeeze().to(self.device)
            return
        assert os.path.exists(path_to_reference_audio)
        wave, sr = soundfile.read(path_to_reference_audio)
        if sr != self.audio_preprocessor.sr:
            self.audio_preprocessor = AudioPreprocessor(input_sr=sr, output_sr=16000, cut_silence=True, device=self.device)
        spec = self.audio_preprocessor.audio_to_mel_spec_tensor(wave).transpose(0, 1)
        spec_len = torch.LongTensor([len(spec)])
        self.default_utterance_embedding = self.style_embedding_function(spec.unsqueeze(0).to(self.device),
                                                                         spec_len.unsqueeze(0).to(self.device)).squeeze()

    def set_language(self, lang_id):
        """
        The id parameter actually refers to the shorthand. This has become ambiguous with the introduction of the actual language IDs
        """
        print("in set_language")
        self.set_phonemizer_language(lang_id=lang_id)
        self.set_accent_language(lang_id=lang_id)

    def set_phonemizer_language(self, lang_id):
        self.text2phone = ArticulatoryCombinedTextFrontend(language=lang_id, add_silence_to_end=True)

    def set_accent_language(self, lang_id):
        if self.use_lang_id:
            self.lang_id = get_language_id(lang_id).to(self.device)
        else:
            self.lang_id = None

    def forward(self,
                text,
                view=False,
                duration_scaling_factor=1.0,
                pitch_variance_scale=1.0,
                energy_variance_scale=1.0,
                pause_duration_scaling_factor=1.0,
                durations=None,
                pitch=None,
                energy=None,
                input_is_phones=False,
                return_plot_as_filepath=False):
        """
        duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
        pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
        energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        print("in forward")
        with torch.inference_mode():
            print("torch inference_mode")
            phones = self.text2phone.string_to_tensor(text, input_phonemes=input_is_phones).to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(phones,
                                                           return_duration_pitch_energy=True,
                                                           utterance_embedding=self.default_utterance_embedding,
                                                           durations=durations,
                                                           pitch=pitch,
                                                           energy=energy,
                                                           lang_id=self.lang_id,
                                                           duration_scaling_factor=duration_scaling_factor,
                                                           pitch_variance_scale=pitch_variance_scale,
                                                           energy_variance_scale=energy_variance_scale,
                                                           pause_duration_scaling_factor=pause_duration_scaling_factor)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)
            print("size of wav")
            print(sys.getsizeof(wave))
        return wave

    def do_tts(self, text_list,
                     duration_scaling_factor=1.0,
                     pitch_variance_scale=1.0,
                     energy_variance_scale=1.0,
                     silent=False,
                     dur_list=None,
                     pitch_list=None,
                     energy_list=None,
                     language="en"):
        """
        Args:
            silent: Whether to be verbose about the process
            text_list: A list of strings to be read
            energy_list: list of energy tensors to be used for the texts
            pitch_list: list of pitch tensors to be used for the texts
            dur_list: list of duration tensors to be used for the texts
            duration_scaling_factor: reasonable values are 0.8 < scale < 1.2.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
            pitch_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
            energy_variance_scale: reasonable values are 0.6 < scale < 1.4.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        if not dur_list:
            dur_list = []
        if not pitch_list:
            pitch_list = []
        if not energy_list:
            energy_list = []
        wav = None
        silence = torch.zeros([10600])
        print(f"set language {language}")
        self.set_language(language)
        for (text, durations, pitch, energy) in itertools.zip_longest(text_list, dur_list, pitch_list, energy_list):
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                if wav is None:
                    print("wav is None")
                    wav = self(text,
                               durations=durations.to(self.device) if durations is not None else None,
                               pitch=pitch.to(self.device) if pitch is not None else None,
                               energy=energy.to(self.device) if energy is not None else None,
                               duration_scaling_factor=duration_scaling_factor,
                               pitch_variance_scale=pitch_variance_scale,
                               energy_variance_scale=energy_variance_scale).cpu()
                    wav = torch.cat((wav, silence), 0)
                else:
                    print("wav is sommething")
                    wav = torch.cat((wav, self(text,
                                               durations=durations.to(self.device) if durations is not None else None,
                                               pitch=pitch.to(self.device) if pitch is not None else None,
                                               energy=energy.to(self.device) if energy is not None else None,
                                               duration_scaling_factor=duration_scaling_factor,
                                               pitch_variance_scale=pitch_variance_scale,
                                               energy_variance_scale=energy_variance_scale).cpu()), 0)
                    wav = torch.cat((wav, silence), 0)
        # wav = [val for val in wav for _ in (0, 1)]  # doubling the sampling rate for better compatibility (24kHz is not as standard as 48kHz)
        print("return wav")
        return wav
        # soundfile.write(file=file_location, data=wav, format='FLAC', samplerate=24000)