from os import path
from pydub import AudioSegment

#source file
src="DrGCWConversation.mp3"
tgt="Conversation.wav"

sound = AudioSegment.from_mp3(src)
sound.export(tgt,format="wav")