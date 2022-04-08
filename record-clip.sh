#!/bin/bash

arecord -f dat --rate=48000 --file-type=wav --duration=15 --device="hw:3,0" --use-strftime ./recordings/deck/%F-%X.wav