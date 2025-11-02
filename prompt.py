from utils import read_txt_prompt

sys_prompt_extract = """You are an AI assistant that extracts query related entity nodes and relation edges from multimodal information sources."""
user_prompt = """
<QUERY>
{query_content}
</QUERY>

<TEXT>
{text_content}
</TEXT>

Given text or image that is potentially relevant to the above query, 
extract fact triples from text or image that are explicitly or implicitly mentioned and relevant to the query provided. 
The extracted fact triples should consists of entity nodes and relation edges. 

Guidelines:
1. Extract other significant entities, concepts, or actors mentioned in the text or image.
2. DO NOT create nodes for relationships or actions, create them as edges.
3. DO NOT create nodes for temporal information like dates, times or years, instead, added them to edges later.
4. Be as explicit as possible in your entity node and relation edge names, using full names.

Output format:
Step 1. For each identified entity node, extract and output the following information:
- entity_name: Name of the entity, use same language as input text.
- entity_description: Short and precise description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_description>)

Step 2. From the entities identified in step 1, identify all relation edges as pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of relation edges, extract the following information:
- source_entity: Name of the source entity, as identified in step 1
- target_entity: Name of the target entity, as identified in step 1
- relation_description: Short and precise explanation as to why you think the source entity and the target entity are related to each other, also include the temporal information here
- relation_names: One or more high-level key words that summarize the overarching nature of the relation, focusing on concepts or themes rather than specific details
Format each relation edge as ("relation"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_name>{tuple_delimiter}<relationship_description>)

Step 3. Split different record with {record_delimiter}. When finished, output {completion_delimiter}

An example output template for 2 entities and 1 relation is given as follow, strictly output as this template, do not output any other words:
("entity"{tuple_delimiter}entity1{tuple_delimiter}description for entity1){record_delimiter}
("entity"{tuple_delimiter}entity2{tuple_delimiter}description for entity2){record_delimiter}
("relation"{tuple_delimiter}entity1{tuple_delimiter}entity2{tuple_delimiter}relation1{tuple_delimiter}description for relation1){completion_delimiter}
"""


user_prompt_query = """
<QUERY>
{query_content}
</QUERY>

Given the above query, decompose the query and extract a query graph that contains entity nodes and relation edges for solving the query. 
You need to represent the graph in triplet format. 
Your extracted triples should consists of entity nodes and relation edges. 
For the key entities nodes for solving the query, and not explicitly mentioned in the query, you can represent them with placeholder entities that suit the node's semantic feature, such as "person1", "person2", "location1", "location2", "date1", "date2" etc.
For the key entities nodes that are explicitly mentioned in the query, mark them as "anchor_entity".

Guidelines:
1. Extract other significant entities, concepts, or actors mentioned in the text or image.
2. DO NOT create nodes for relationships or actions, create them as edges.
3. DO NOT create nodes for temporal information like dates, times or years, instead, added them to edges later.
4. Be as explicit as possible in your entity node and relation edge names, using full names.

Output format:
Step 1. For each identified entity node, extract and output the following information:
- entity_name: Name of the entity, use same language as input text.
- entity_description: Short and precise description of the entity's attributes and activities
Format entity as ("entity"{tuple_delimiter}<entity_name>), and the anchor entities should be marked as ("anchor_entity"{tuple_delimiter}<entity_name>).

Step 2. From the entities identified in step 1, identify all relation edges as pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of relation edges, extract the following information:
- source_entity: Name of the source entity, as identified in step 1
- target_entity: Name of the target entity, as identified in step 1
- relation_description: Short and precise explanation as to why you think the source entity and the target entity are related to each other, also include the temporal information here
- relation_names: One or more high-level key words that summarize the overarching nature of the relation, focusing on concepts or themes rather than specific details
Format each relation edge as ("relation"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_name>)

Step 3. Split different record with {record_delimiter}. Remember to firstly output entities, then relations. When finished, output {completion_delimiter}

An example output template for the query "Where was the performer of Count 'Em 88 born" is given as follow, strictly output as this template, do not output any other words:
("anchor_entity"{tuple_delimiter}Count 'Em 88){record_delimiter}
("entity"{tuple_delimiter}person1){record_delimiter}
("entity"{tuple_delimiter}location1){record_delimiter}
("relation"{tuple_delimiter}Count 'Em 88{tuple_delimiter}person1{tuple_delimiter}performed by){record_delimiter}
("relation"{tuple_delimiter}performer{tuple_delimiter}location1{tuple_delimiter}born in){completion_delimiter}
"""




deepseek_prompt = ('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
                   'The assistant first thinks about the reasoning process in the mind and then provides the user '
                   'with the answer. The reasoning process and answer are enclosed within '
                   '<think> </think> and <answer> </answer> tags, respectively, i.e., '
                   '<think> reasoning process here </think> <answer> answer here </answer>. User: prompt. Assistant:')

mm_prompt = ('You are an multi-modeled question answering assistant. '
             'When the user ask you questions, '
             'you need to first think about the reasoning process in the mind, and break the reasoning proces into '
             'reasoning steps (sub-questions) that can be answered with text or image modality side knowledge, '
             'or can be directly answered without any additional side knowledge.  '
             'Try to address the reasoning chain and solve the question step-by-step, '
             'response the user with your reasoning chain and the final answer. \n\n'
             'The reasoning process and answer are enclosed within the following tags: \n'
             '<think> Contains the reasoning process here. </think> \n'
             '<answer> Contains the final answer here, '
             'if you need side information for the final answer, you may skip this tag. </answer> \n'
             '<text_retrieval> Contains the description of the needed text knowledge. </text_retrieval> \n'
             '<image_retrieval> Contains the description of the needed image knowledge. </image_retrieval> \n\n'
             'Example 1: \n'
             'The user is a 23-year-old pregnant woman at 22 weeks gestation with dysuria that\'s getting worse despite home remedies. '
             'She doesn\'t have fever or CVA tenderness, so it\'s likely cystitis. I need to choose the best antibiotic considering pregnancy safety.\n'
             'First, I remember that some antibiotics are unsafe during pregnancy. '
             'Let me think: fluoroquinolones like ciprofloxacin are generally avoided because they can affect fetal cartilage development. '
             'Doxycycline is a tetracycline, which causes tooth discoloration and bone issues in the fetus, so that\'s out.\n'
             'Ampicillin used to be common, but I think E. coli has high resistance rates now. '
             'Maybe it\'s not first-line anymore unless susceptibility is confirmed. Ceftriaxone is a cephalosporin, '
             'safe in pregnancy, but isn\'t that usually for more severe infections or when oral meds can\'t be used? Since she\'s stable, maybe an oral option is better.\n'
             'Nitrofurantoin comes to mind as a UTI treatment. Wait, but isn\'t there a concern with use in the third trimester? '
             'Some risk of hemolytic anemia in the newborn if used near delivery. But she\'s 22 weeks, second trimester, so maybe it\'s okay. '
             'However, I should check guidelines—ACOG or IDSA recommendations. Also, is nitrofurantoin contraindicated in early pregnancy? '
             'I think some sources say avoid in first trimester due to theoretical risks, but second trimester might be safe.\n'
             'Ceftriaxone is injectable, which isn\'t ideal for uncomplicated cystitis. Ampicillin\'s resistance is a problem. '
             'So nitrofurantoin might be the best, but I need to confirm that it\'s recommended in second trimester by current guidelines. '
             'Also, are there alternatives like cephalexin? But the options given don\'t include that. '
             'Between the listed choices, nitrofurantoin and ceftriaxone are possible, but ceftriaxone is more for severe cases. '
             'Need to verify the recommended first-line treatment for uncomplicated UTI in pregnancy from a reliable source.'
             '</think>\n'
             '<text_retrieval> Confirm contraindications (e.g., fluoroquinolones, tetracyclines) '
             'and trimester-specific risks (e.g., nitrofurantoin in first vs. second trimester) of Antibiotic Safety in Pregnancy'
             'from FDA pregnancy categories, ACOG guidelines, UpToDate/DynaMed entries, drug monographs (Micromedex, LactMed).. '
             '</text_retrieval> \n'
             '<image_retrieval> Confirm contraindications (e.g., fluoroquinolones, tetracyclines) '
             'and trimester-specific risks (e.g., nitrofurantoin in first vs. second trimester) of Antibiotic Safety in Pregnancy'
             'from Infographics/tables summarizing antibiotic safety in pregnancy '
             '(e.g., from medical textbooks or guideline appendices). '
             '</image_retrieval> \n'
             '<text_retrieval> Need regional resistance rates to ampicillin for empiric UTI treatment from Local antibiograms '
             '(hospital reports), IDSA/CDC guidelines to confirm. '
             'Resistance Patterns of E. coli. '
             '</text_retrieval> \n'
             '<image_retrieval> Need regional resistance rates to ampicillin for empiric UTI treatment '
             'from Resistance trend graphs in public health reports (e.g., CDC annual reports) to confirm. '
             'Resistance Patterns of E. coli.'
             '</image_retrieval> \n'
             '<text_retrieval> Need first-line antibiotics per ACOG/IDSA (e.g., nitrofurantoin, cephalexin) '
             'from ACOG Practice Bulletins, IDSA UTI guidelines to confirm Guideline Recommendations for Uncomplicated Cystitis.'
             '</text_retrieval> \n'
             '<image_retrieval> Need first-line antibiotics per ACOG/IDSA (e.g., nitrofurantoin, cephalexin) '
             'from algorithm flowcharts in guideline documents (e.g., "UTI in Pregnancy" decision trees) to confirm Guideline Recommendations for Uncomplicated Cystitis.'
             '</image_retrieval> \n'
             '<text_retrieval> About Nitrofurantoin Safety in Second Trimester, '
             'need to confirm safe use at 22 weeks gestation (avoidance in first trimester/near term) from Drug monographs (Micromedex).'
             '</text_retrieval> \n'
             '<text_retrieval> About Ceftriaxone Indications, '
             'need to confirm whether it’s reserved for severe infections (e.g., pyelonephritis) '
             'or inappropriate for uncomplicated cystitis. Source: IDSA guidelines, antibiotic stewardship documents.'
             '</text_retrieval> \n'
             '<image_retrieval> About Ceftriaxone Indications, '
             'need to confirm whether it’s reserved for severe infections (e.g., pyelonephritis) '
             'or inappropriate for uncomplicated cystitis. '
             'Source: Antibiotic spectrum charts (e.g., "UTI Treatment Options" tables).'
             '</image_retrieval> \n'
             'Real Data: \n'
             'User: {query}\n'
             'Assistant:'
             )

system_prompt = '\n'.join(read_txt_prompt('./prompt/system_prompt.txt'))

system_prompt_SQL = '\n'.join(read_txt_prompt('./prompt/system_prompt_SPARQL.txt'))

system_prompt_ner = '\n'.join(read_txt_prompt('./prompt/system_prompt_ner.txt'))

system_prompt_cot = '\n'.join(read_txt_prompt('./prompt/system_prompt_cot.txt'))

# system_prompt_ent = '\n'.join(read_txt_prompt('system_prompt_ent.txt'))

system_prompt_mramg="""
# Task
Imagine you are a text QA expert, skilled in delivering contextually relevant answers. You will receive:   
1. Query.  
2. Contexts. 

Your task is to answer the query based solely on the content of the context. 

# Requirements    
Ensure that your answer does not include any additional information outside the context. Please note that your answer should be in pure text format.

# Output Format    
Provide the answer in pure text format. Answer the question in <ANS> Answer here </ANS>. Do not include any information beyond what is contained in the context.
"""