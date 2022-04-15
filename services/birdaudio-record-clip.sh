#!/bin/bash

echo 'Recording audio...'
arecord -f dat --rate=48000 --file-type=wav --duration=15 --device="hw:1,0" --use-strftime ~/birdaudio/recordings/deck/%F-%X.wav