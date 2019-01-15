for file in ../gifs/*.gif
do convert $file \( +clone -set delay 500 \) +swap +delete  $file
done