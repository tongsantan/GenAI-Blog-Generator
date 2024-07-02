import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text,no_words,blog_style):

    ### LLama2 model
    llm=CTransformers(model='./model/llama-2-7b-chat.ggmlv3.q5_1.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template

    template="""
        Write a blog topic on {input_text} for target audience {blog_style}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    prompt_value = prompt.invoke({"blog_style": blog_style, "input_text": input_text, "no_words": no_words})
    response = llm.invoke(prompt_value)
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs")

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields

col1,col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for',
                            ('Researchers', 'Children', 'General Public'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))