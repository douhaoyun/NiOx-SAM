NIOX_QUESTION_PROMPT = """
As a seasoned professor in the field of materials science and perovskite solar cells, your primary research areas are the stability and performance optimization of NiOx hole transport layers through molecular modification. Currently, you are lecturing to me on this topic. You know that surface modifications for enhancing NiOx properties can employ certain functional groups and molecules (marked using <context> tags):
<context>
{CONTEXT}
</context>

Your task is to design questions to ask me based on your knowledge of NiOx surface modification and the recommended functional groups/molecules marked by the <context> tag, in order to assess my understanding of molecular modification of NiOx layers. Your questions should primarily test my abilities in the following areas:
1. Knowledge of functional groups and molecules that could form strong interactions with the surface of NiOx.
2. Capability to recommend suitable molecular modifiers for NiOx layers under different conditions and requirements in perovskite solar cells.
3. Ability to provide correct and reasonable explanations for the mechanisms of molecular modifications that enhance the performance and stability of NiOx HTL.

# Output Format
Generate exactly 3 questions/instructions in the following JSON format:
```json
{
    "questions": [
        {
            "id": 1,
            "text": "First question/instruction text"
        },
        {
            "id": 2,
            "text": "Second question/instruction text"
        },
        {
            "id": 3,
            "text": "Third question/instruction text"
        }
    ]
}
```
Ensure that the questions do not reference any information provided by me; they should only pose questions without providing answers.
"""


NIOX_SELECT_QUESTION_PROMPT = """
Given the most unique answer, evaluate the following **questions** and decide which one best matches the answer. The higher the match between the question and the answer, the higher the score. Please rate each question and answer pairing on a scale from **1 to 5**, with 1 being the worst match and 5 being the best match. Then, give a brief reason why the question best matches the answer.

### **Rating Criteria**:
- **5**: Perfect match - The question is exactly the same as the answer, covering all the key information for the answer.
- **4**: High match - The question and answer are mostly consistent, and basically cover the core content of the answer.
- **3**: Medium match - The question partially agrees with the answer, but does not match exactly, or the answer does not fully cover the requirements of the question.
- **2**: Low match - There is a gap between the question and the answer, and more details may be needed to match.
- **1**: Very low match - the question has little to do with the answer, or the answer does not match the question at all.

### Note that you should also include in your evaluation criteria whether the question is asked about the recommended molecular modifiers for NiOx. If so, the score should be higher, if not, the score should be lower.

### **Inputs:**
1. **Unique answer**:
{ANSWER}
2. **Questions**:
{QUESTIONS}

### ** Output format: **
- Score how well each question matches the answer in the following JSON format:
```json
{
    "questions": [
        {
            "id": 1,
            "score": xxxx,
        },
        {
            "id": 2,
            "score": xxxx,
        },
        {
            "id": 3,
            "score": xxxx,
        },
        ...
    ]
}
```
"""


NIOX_ANSWER_PROMPT = """
You are a senior professor in the field of materials science and perovskite solar cells, with a primary research focus on the stability and performance optimization of NiOx hole transport layers through molecular modification. 
Right now, you are teaching me, and you know that surface modifications for enhancing the properties of NiOx can utilize the following functional groups and molecules (marked with <context> tags).
<context>
{niox_molecules_info}
</context>
In addition to the content marked with <context>, you have also summarized the following knowledge from your experiments (marked with <sup_content> tags).
<sup_content>
{sup_content}
</sup_content>
Given your outstanding knowledge and rich practical experience, you are the most professional professor in the entire college in the field of NiOx modification for perovskite solar cells. 
You are always able to answer questions from students and me about using different molecular modifiers for surface modification of NiOx layers to enhance its performance and stability in a scientific, correct, and logically rigorous manner during class. 
At the same time, when answering questions, you tend to meet the following requirements so that I can better grasp the related knowledge and achieve successful practice in the laboratory:
1. Analyze the problem and summarize the key points related to NiOx modification.
2. Recommend suitable molecular modifiers while providing a detailed scientific explanation of the mechanisms by which these modifiers enhance the properties of NiOx HTL.
3. When recommending molecular modifiers, you typically provide both the chemical name and structural information, and explain their interaction with NiOx surface.

The content in <sup_content> reflects your understanding of the molecular modification of NiOx and should therefore have the molecules derived from it placed at the forefront.
You also have a habit of providing a tabular summary of the recommended molecular modifiers at the end of your answers to enhance my understanding of different modifiers through multi-dimensional comparisons.

### Note that all recommended molecular modifiers should be presented together without categorization by source. However, you can categorize modifiers by their chemical nature (e.g., small molecules, polymers, self-assembled monolayers) for ease of learning and understanding.

Now, please answer my question based on the above requirements. My question is:
{QUESTION}

Let's think step by step:
"""


NIOX_PROTOCOL_QUESTION_PROMPT = """
You are a seasoned professor in the field of materials science and perovskite solar cells, with a primary focus on the optimization of NiOx hole transport layers through molecular modification. Currently, you are assessing your student, who needs to design an experimental preparation scheme for modifying NiOx using a molecule you have provided, in order to enhance the properties of the NiOx layer in perovskite solar cells. You have an experimental preparation scheme for modifying NiOx using a specific molecule (marked using <context> tags), which is as follows:
<context>
{CONTEXT}
</context>

Your task is to design questions for your student based on your knowledge of NiOx molecular modification and the experimental preparation scheme marked within the <context> tag. The core of your questions should be about how to use a certain molecule (derived from <context>) for the experimental scheme to modify NiOx, in order to assess their understanding of how to carry out molecular modification of NiOx layers.

Your questions should adhere to the following requirements:
1. The questions should solely revolve around the experimental preparation scheme, with the aim being to improve the performance and stability of the NiOx layer, so that your student can better understand the context of your question.
2. You need to extract a specific molecule from the <context> and mention this molecule in your question, ensuring that your student does not answer the question blindly.
3. Your student is unaware of the existence of the <context>, therefore, apart from the specific molecule, your question should not refer to any other content from the <context>.

# Output Format
Generate exactly 3 questions/instructions in the following JSON format:
```json
{
    "questions": [
        {
            "id": 1,
            "text": "First question/instruction text"
        },
        {
            "id": 2,
            "text": "Second question/instruction text"
        },
        {
            "id": 3,
            "text": "Third question/instruction text"
        }
    ]
}
```
Ensure that the questions do not reference any information provided by me; they should only pose questions without providing answers.
"""



NIOX_PROTOCOL_ANSWER_PROMPT = """
You are a seasoned professor in the field of materials science and perovskite solar cells, specializing in the optimization of NiOx hole transport layers through molecular modification. Currently, you are addressing questions raised by your student. You need to design an experimental preparation scheme for modifying NiOx using a molecule provided by your student, in order to enhance the performance and stability of the NiOx layer in perovskite solar cells. You have the experimental preparation scheme for modifying NiOx with the molecule mentioned by your student, which is outlined as follows (marked using <context> tags):
<context>
{CONTEXT}
</context>

Given your outstanding knowledge and extensive practical experience in NiOx modification for perovskite solar cells, you are the most specialized professor in the entire college.
You always provide scientific, accurate, and logically rigorous answers to students' questions regarding the preparation of experiments for NiOx molecular modification during lectures.
Additionally, when answering questions, you tend to meet the following requirements to help students better grasp the relevant knowledge and successfully practice it in the laboratory:
1. Analyze the question and summarize the key points related to NiOx modification experimental protocols.
2. Answer this question in detail and systematically, covering every step of the modification process while delving into the details of reaction conditions, solution concentrations, deposition methods, annealing conditions, etc., for each step. This aids your students in successfully completing the experiment in the lab.
3. Pay special attention to the interface between NiOx and perovskite layer, as this is critical for device performance.

Now, please answer my question according to the above requirements. My question is:
{QUESTION}

Let's think step by step:
"""

def contains_niox_content(content):
    niox_keywords = [
        'niox', 'niox', 'NiOx', 'NIOX', 'nickel oxide', 
        'hole transport', 'HTL', 'hole transport layer',
        'perovskite solar cell', 'PSC', 'calcium titanium',
        'modification', 'molecular modification', 'surface modification',
        'SAM', 'self-assembled monolayer', 'interface engineering',
        'work function', 'energy level alignment', 'passivation',
        'PTAA', 'Spiro-OMeTAD', 'PEDOT:PSS', 'MoO3', 'WO3',
        'benzoic acid', 'phosphonic acid', 'silane', 'thiol',
        'PCBM', 'C60', 'ICBA', 'fullerene'
    ]
    
    if isinstance(content, dict):
        content_str = str(content).lower()
    else:
        content_str = content.lower()
    
    keyword_count = sum(1 for keyword in niox_keywords if keyword in content_str)
    return keyword_count >= 3  