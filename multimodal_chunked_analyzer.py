#!/usr/bin/env python3
"""
Multimodal Video Analyzer
-------------------------
Analiza videos locales o de YouTube en chunks configurables, aplicando:
- ASR (Whisper)
- Emociones de audio y texto
- Clasificación de tópicos
- Edad/Género por audio
- Análisis facial por video
- Normalización USS (VADER + pysentimiento + prosodia)
- Diarización de hablantes (pyannote.audio)

Salida: JSON con resumen global y resultados por chunk.
"""

import os, sys, time, json, math, shutil, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import moviepy.editor as mpy
from tqdm import tqdm

MAIN_OUTPUT_FOLDER = "outputs"

# ---------------------------
# Utilidades
# ---------------------------
def sec_to_hhmmss(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def unique_output_folder(base_name: str) -> str:
    """Crea carpeta única: base_name, base_name_2, ..."""
    base_name = MAIN_OUTPUT_FOLDER + "/" + base_name
    folder = Path(base_name)

    counter = 1
    while folder.exists():
        folder = Path(f"{base_name}_{counter}")
        counter += 1

    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


# ---------------------------
# Analizadores (clases simplificadas)
# ---------------------------
class ASRWhisper:

    def __init__(self, model_name="large-v3"):
        import whisper
        self.model = whisper.load_model(model_name)

    def transcribe(self, wav_path, language=None):
        opts = {}
        if language: opts["language"] = language
        return self.model.transcribe(wav_path, **opts, verbose=False)


class AudioSER:

    def __init__(self, model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        from transformers import pipeline, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.pipe = pipeline("audio-classification", model=model, feature_extractor=fe, top_k=None)
        self.mapping = {
            "enojo":["ang"],
            "alegría":["happy","joy"],
            "miedo":["fear"],
            "tristeza":["sad"],
            "sorpresa":["surprise"]
        }

    def predict(self, wav):
        try:
            res = self.pipe(wav)
            out = {k:0.0 for k in self.mapping}

            for o in res:
                lab, sc = o["label"].lower(), o["score"]
                for tgt,kws in self.mapping.items():
                    if any(kw in lab for kw in kws): out[tgt]+=sc

            return {k: round(min(v*10,10),3) for k,v in out.items()}

        except:
            return {k:0.0 for k in self.mapping}


class TextEmotion:

    def __init__(self, model="SamLowe/roberta-base-go_emotions"):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model, return_all_scores=True)
        self.map = {"enojo":["anger"],"alegría":["joy"],"miedo":["fear"],"tristeza":["sadness"],"sorpresa":["surprise"]}

    def predict(self, text):
        if not text.strip():
            return {k:0.0 for k in self.map}

        res = self.pipe(text)[0]
        out = {k:0.0 for k in self.map}

        for r in res:
            for tgt,kws in self.map.items():
                if any(kw in r["label"].lower() for kw in kws): out[tgt]+=r["score"]

        return {k: round(min(v*10,10),3) for k,v in out.items()}


class TopicClassifier:

    def __init__(self, model="facebook/bart-large-mnli"):
        from transformers import pipeline
        self.pipe = pipeline("zero-shot-classification", model=model)
        self.labels = [
            "política",
            "educación",
            "historia",
            "tecnología",
            "ciencia",
            "deporte",
            "economía",
            "religión",
            "otro"
        ]

    def predict(self, text):
        if not text.strip():
            return {t:0.0 for t in self.labels}
        res = self.pipe(text, self.labels, multi_label=True)
        return {l: round(s*10,3) for l,s in zip(res["labels"], res["scores"])}


class AgeGenderAudio:

    def __init__(self, model="audeering/wav2vec2-large-robust-24-ft-age-gender"):
        from transformers import pipeline, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model)
        model_m = AutoModelForAudioClassification.from_pretrained(model)
        self.pipe = pipeline("audio-classification", model=model_m, feature_extractor=fe, top_k=None)

    def predict(self, wav):
        try: out = self.pipe(wav)
        except: return {
            "gender":"desconocido",
            "age_raw":"desconocido"
        }
        g="desconocido"
        age="desconocido"
        for o in out:
            if "male" in o["label"].lower(): g="hombre"
            elif "female" in o["label"].lower(): g="mujer"
            else: age=o["label"]
        return {
            "gender":g,
            "age_raw":age
        }


class VideoAnalyzer:

    def __init__(self):
        try:
            from deepface import DeepFace
            import cv2
            self.DeepFace, self.cv2, self.available = DeepFace, cv2, True
        except: self.available=False

    def analyze(self, video_path, timestamps):
        if not self.available: return []
        cap = self.cv2.VideoCapture(video_path)
        fps = cap.get(self.cv2.CAP_PROP_FPS) or 25
        out=[]
        for t in timestamps:
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, int(t*fps))
            ok, frame = cap.read()
            if not ok: continue
            try: an = self.DeepFace.analyze(frame, actions=["age","gender"], enforce_detection=False)
            except: continue
            it = an[0] if isinstance(an,list) else an
            out.append({
                "gender_video":it.get("dominant_gender"),
                "age_raw_video":it.get("age")}
            )
        cap.release()
        return out


class USSNormalizer:

    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        import pysentimiento
        self.vader = SentimentIntensityAnalyzer()
        self.pysent = pysentimiento.create_analyzer(task="sentiment", lang="es")

    def vader_score(self, text):
        if not text: return 0.0
        return self.vader.polarity_scores(text)["compound"]

    def pysent_score(self, text):
        if not text: return 0.0
        res = self.pysent.predict(text)
        if res.output=="POS": return res.probas["POS"]
        if res.output=="NEG": return -res.probas["NEG"]
        return 0.0

    def prosody_score(self, wav):
        import librosa
        try:
            y,sr=librosa.load(wav,sr=16000)
            rms=librosa.feature.rms(y=y).mean()
            pitch=librosa.yin(y,50,400,sr=sr).mean()
            tempo,_=librosa.beat.beat_track(y=y,sr=sr)
            score=(rms/0.1+pitch/200+tempo/120)/3
            return min(max(score,0.0),1.0)
        except: return 0.0

    def normalize_to_uss(self, text, wav):
        v,p,pr=self.vader_score(text),self.pysent_score(text),self.prosody_score(wav)
        raw=0.4*v+0.4*p+0.2*pr
        uss=round(((raw+1)/2)*10,1)
        return {
            "vader":v,
            "pysent":p,
            "prosody":pr,
            "USS":uss
        }


class SpeakerDiarizer:

    def __init__(self, token=None):
        from pyannote.audio import Pipeline
        self.pipeline = None
        try:
            if not token:
                raise RuntimeError("Falta --hf_token para diarización.")
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
        except Exception as e:
            print("⚠️ No se pudo cargar diarización:", e)
            self.pipeline = None

    def run(self, wav, whisper_segments):
        if self.pipeline is None:
            return []
        diar = self.pipeline(wav)
        out = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            segs = [ws["text"] for ws in whisper_segments if ws["start"] >= turn.start and ws["end"] <= turn.end]
            out.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "text": " ".join(segs)}
            )
        return out


# ---------------------------
# Función de análisis de chunk
# ---------------------------
def analyze_chunk(wav_path, video_path, st, et, analyzers, config):
    chunk={
        "segment":f"{int(st)}_{int(et)}",
        "time":f"{sec_to_hhmmss(st)}-{sec_to_hhmmss(et)}"
    }

    with ThreadPoolExecutor(max_workers=config.get("max_workers",6)) as ex:
        f_asr=ex.submit(analyzers["asr"].transcribe,wav_path,config.get("language"))
        f_aud=ex.submit(analyzers["audio_ser"].predict,wav_path)
        f_age=ex.submit(analyzers["aga"].predict,wav_path)

        if analyzers["video"].available:
            f_vid=ex.submit(analyzers["video"].analyze,video_path,[st+(et-st)/2])
        else:
            f_vid=None

        tr=f_asr.result()
        txt=tr.get("text","")
        whisper_segments=tr.get("segments",[])
        f_txt=ex.submit(analyzers["text_emotion"].predict,txt)
        f_top=ex.submit(analyzers["topic"].predict,txt)
        audio,age,text,topic=f_aud.result(),f_age.result(),f_txt.result(),f_top.result()
        video=f_vid.result() if f_vid else []

    diar=analyzers["diar"].run(wav_path,whisper_segments)
    uss=analyzers["uss"].normalize_to_uss(txt,wav_path)

    chunk.update({
        "transcription":txt,
        "speaker_segments":diar,
        "audio_emotions":audio,
        "text_emotions":text,
        "topics":topic,
        "age_gender_audio":age,
        "video_detections":video,
        "USS_details":uss,
        "USS_final":uss["USS"]
    })

    return chunk

# ---------------------------
# Proceso general
# ---------------------------
def process(media_path,is_video,out_dir,seg_dur,whisper_model,language,config,source_url=None):
    safe_mkdir(out_dir)
    tmp=os.path.join(out_dir,"tmp_chunks")
    shutil.rmtree(tmp,ignore_errors=True)
    safe_mkdir(tmp)
    clip=mpy.VideoFileClip(media_path) if is_video else mpy.AudioFileClip(media_path)
    dur=clip.duration
    nseg=math.ceil(dur/seg_dur)
    analyzers={
        "asr":ASRWhisper(whisper_model),
        "audio_ser":AudioSER(),
        "text_emotion":TextEmotion(),
        "topic":TopicClassifier(),
        "aga":AgeGenderAudio(),
        "video":VideoAnalyzer(),
        "uss":USSNormalizer(),
        "diar":SpeakerDiarizer(token=config.get("hf_token")),
    }
    results=[]
    t0=time.time()
    for i in tqdm(range(nseg),desc="Chunks"):
        st,et=i*seg_dur,min((i+1)*seg_dur,dur)
        wav=os.path.join(tmp,f"seg{i:03d}.wav")
        sub=clip.subclip(st,et)
        if sub.audio: sub.audio.write_audiofile(wav,fps=16000,logger=None)
        else: open(wav,"wb").close()
        results.append(analyze_chunk(wav,media_path,st,et,analyzers,config))
    total=time.time()-t0
    overall={
        "source": source_url if source_url else media_path,
        "segments":len(results),
        "duration_seconds":dur,
        "total_processing_time_seconds":round(total,2),
        "chunks":results
    }
    with open(os.path.join(out_dir,"output.json"),"w",encoding="utf-8") as f: json.dump(overall,f,ensure_ascii=False,indent=2)
    return overall

# ---------------------------
# CLI
# ---------------------------
def main():
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input","-i")
    g.add_argument("--url","-u")
    p.add_argument("--output_folder","-of",default="output_analysis",help="Carpeta base de salida")
    p.add_argument("--segment","-s",type=int,default=30)
    p.add_argument("--whisper_model","-wm",default="large-v3")
    p.add_argument("--language","-lang",default=None)
    p.add_argument("--max_workers",type=int,default=6)
    p.add_argument("--hf_token",type=str,help="HuggingFace token para diarización")
    args=p.parse_args()

    out_dir = unique_output_folder(args.output_folder)

    if args.url:
        import yt_dlp
        ydl=yt_dlp.YoutubeDL({"outtmpl":os.path.join(out_dir,"yt.%(ext)s")})
        info=ydl.extract_info(args.url,download=True)
        media_path=ydl.prepare_filename(info)
        is_video=True
        source_url=args.url
    else:
        media_path=args.input
        is_video=True
        source_url=None

    config={
        "max_workers":args.max_workers,
        "language":args.language,
        "hf_token":args.hf_token
    }

    overall=process(media_path,is_video,out_dir,args.segment,args.whisper_model,args.language,config,source_url=source_url)

    print(json.dumps({
        "source":overall["source"],
        "segments":overall["segments"],
        "time_s":overall["total_processing_time_seconds"]
    },indent=2,ensure_ascii=False))

if __name__=="__main__": main()
