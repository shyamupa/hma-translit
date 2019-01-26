-Code for the EMNLP paper, "Bootstrapping Transliteration with Guided Discovery for Low-Resource Languages".

## Running the code

1. First compile the C code for the aligner.
```bash
cd baseline/
make
```

2. write you train, dev and test data in the following format, 

```
x1 x2 x3<tab>y1 y2 y3 y4 y5
```
where `x1x2x3` is the input word (`xi` is the character), and `y1y2y3y4y5` is the desired output (transliteration). Example train and test files for bengali are in data/ folder. There is a optional 3rd column marking whether the word is *native* or *foreign* (see the paper for these terms); this column can be ignored for most purposes. 


3. Run `train_model_on_files.sh` on your train (say train.txt) and dev file (dev.txt) as follows,

```
./train_model_on_files.sh train.txt dev.txt 100 translit.model
```

where 100 is the random seed and translit.model is the output model. Other parameters(see `utils/arguments.py` for options) can be specified by modifying the `train_model_on_files.sh` script appropriately.

4. Test the trained model as follows,

```
./load_and_test_model_on_files.sh train.txt test.txt translit.model 100 output.txt
```

The output should report relevant metrics,

```
...
...
:INFO: --------------------TEST--------------------
:INFO: running infer on example 200
:INFO: running infer on example 400
:INFO: running infer on example 600
:INFO: running infer on example 800
:INFO: accuracy 367/997=0.37
:INFO: accuracy (nat) 308/661=0.47
:INFO: accuracy (eng) 59/336=0.18
:INFO: ********************total********************
:INFO: ACC:          0.371457 (367/988)
:INFO: Mean F-score: 0.910995
:INFO: Mean ED@1: 1.136640+-1.167
:INFO: Mean NED@1: 0.084884
:INFO: Median ED@1: 1.000000
...
...
```

There is also a interactive mode where one can input test words directly,

```
./load_and_test_model_interactive.sh <ftrain> <model> <seed>
...
...
:INFO: => loading checkpoint hindi.model
:INFO: => loaded checkpoint!
enter surface:ओबामा
ओ ब ा म ा
[(-0.4624647759074629, 'o b a m a')]
```

### Citation

```
@InProceedings{UKR18,
  author =       {Upadhyay, Shyam and Kodner, Jordan and Roth, Dan},
  title =        {Bootstrapping Transliteration with Guided Discovery for Low-Resource Languages},
  booktitle =    {EMNLP},
  year =         {2018},
}
```