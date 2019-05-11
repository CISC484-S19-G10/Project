import music21
from PIL import Image, ImageDraw
from pprint import pprint
import os


#s = music21.converter.parse("./boop.mid")

DRUM_MIDI_START=35
DRUM_MIDI_END=81

def note_to_pixel(note):
    #print(note.pitch.midi)
    return beat_to_pixel(note.offset), note.pitch.midi-DRUM_MIDI_START

def beat_to_pixel(offset):
    return int(float(offset*10))


WHITE = (255,255,255)

max_beats = 8*10

def valid_pixel(pixel):
    return pixel[0]<max_beats and pixel[0]>=0 and pixel[1]>=0 and pixel[1]<DRUM_MIDI_END

def convert(base, name, pixel_map):

    chordIter = base.iter
    noteIter = base.iter

    chordIter.getElementsByClass('Chord')
    noteIter.getElementsByClass('Note')

        

    for note in noteIter:
        pixel = note_to_pixel(note)
        #print(pixel)
        if valid_pixel(pixel):
            pixel_map[pixel[0],pixel[1]] = WHITE

    for chord in chordIter:
        #print(chord)
        offset = beat_to_pixel(chord.beat)
        for pitch in chord.pitches:
            y = pitch.midi-DRUM_MIDI_START
            pixel = (offset,y)
            if valid_pixel(pixel):
                pixel_map[pixel[0],pixel[1]] = WHITE

    #img.show('black.png')



def midi2img(filename, name):
    try:
        s = music21.converter.parse(filename)
    #except:
        
    
        for part in s:
            #print(filename)
            #print(part)
            #for thing in part:
                #print(thing)

            voice_iter = part.iter
            voice_iter.getElementsByClass('Voice')
            img = Image.new('RGB', (max_beats,DRUM_MIDI_END), color = 'black')
            pixel_map = img.load()

            voice_count = 0
            for voice in voice_iter:
                convert(voice, name,pixel_map)
                voice_count+=1
                #print("#")

            if voice_count == 0:
                convert(part,name,pixel_map)

            img.save(filename+".png")
    except:
        print("Broke")


def open_folder():
    for root, dirs, files in os.walk("./gen_midi"):
        for f in files:
            path = os.path.join(root,f)

            png_filename = os.path.basename(path).split(".")[0]+".mid.png"
            #print(f.split("."))
            png_filename = os.path.join(root,png_filename)
            #print(f.split(".")[1])
            exists = os.path.isfile(png_filename)

            if(not exists):
                if (f.split(".")[-1] == "mid"):
                    print(path)
                    midi2img(path,f)
                else:
                    print("Skipping")
            #else:
                #print("File exists")



open_folder()
#midi2img("./why.mid", "work")

    



