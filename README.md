# fantasy-name-classifier

Classify elvish and dwarfish names. See [here](https://www.reddit.com/r/dailyprogrammer_ideas/comments/9o9vtj/hard_tell_elf_names_and_dwarf_names_apart/)

## Install

Tested on Python 3.7.

```
pip install -r requirements.txt
```

## Run

Train model on initial data:

```
python train.py data/train.txt
```

This will serialize a model into the `data` directory for later use.

We can now run this model over any input file in the correct format.

```
python test.py data/validate.txt
python test.py data/final.txt
```

## Notes

With the existing seeded classifiecation, scores 0.77 and 0.76 on the non-training inputs.

Best features appear to be:

- position of some letters (E, I, L)
- value of the final few characters
- where double-letter groups are in the name
