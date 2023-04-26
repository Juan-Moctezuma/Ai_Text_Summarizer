######### PART 1 - Importing Libraries
import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer # Pegasus Method

######### PART 2 - SET UP MODELS
# Load Pegasus tokenizer 
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
# Load Pegasus model 
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

######### PART 3 - APPLICATION FRONT-END UPPER SECTION
# Title
st.title("Text Summarization")

######### PART 4 - ENTER TEXT
text_input = st.text_area("Enter large corpus of text in here: ", height=300)
submit = st.button('Get Summary')

######### PART 5 - BACKEND
if submit and text_input is not None:
    try:
        with st.spinner("Ai will compile your results shortly..."): 
            # Create tokens - number representation of our text
            tokens = tokenizer(text_input, truncation = True, padding = "longest", return_tensors = "pt")
            # Summarize 
            summary = model.generate(**tokens)
            # Decode summary
            x = tokenizer.decode(summary[0])
            st.success("Your Text is Ready")
            st.write(x)
    except:
        st.markdown("Error - Your Text can't be summarized or it's not in english language")
