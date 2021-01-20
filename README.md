# Seq2Seq-Poem-Generation

This seq2seq model is implemented by tensorflow(1.11.0 or 1.12.0) to generate acrostic poems. Acrostic poems is where you use the first letter to spell a word or phrase.

# Example
The dataset contains around 130,000 Chinese poems and it is in `Data` folder. For better understanding, below is an example of English acrostic poem.

Given input word __FALL__, the output of the model would be a poem like below:

**F**armers bring in the harvest from the land.  
**A**nimals prepare for the winter chill.  
**L**eaves fall from the trees -- floating  
**L**ightly to the ground.  

# Usuage

STEP 1: 
Preprocess data and generate:  
>>`data.src.train`  
>>`data.trg.train`  
>>`data.src.valid`  
>>`data.trg.valid`  
>>`data.src.dict`  
>>`data.trg.dict`  

STEP 2: Run `run.sh` 

STEP 3: Run `generate.sh`

For better generation result, one may consider use beamsearch instead of greedy decoder.