import os

import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import uuid
from PIL import Image
import requests
from io import BytesIO
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
import faiss


class VanillaRAG:
    def __init__(self, args):
        self.vector_store = None
        self.embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.image_caption_client = OpenAI(
            base_url='http://localhost:8001/v1',
            api_key='EMPTY',
        )
        models = self.image_caption_client.models.list()
        self.image_caption_model = models.data[0].id if models.data else "Qwen/Qwen2.5-VL-7B-Instruct"
        self.args = args
        self.index_path = f'./working_dir_{self.args.dataset}'
        if not os.path.exists(self.index_path):
            os.mkdir(self.index_path)

    def index_from_doc(self, docs):
        # faiss_index = faiss.IndexFlatL2(768)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.embed_model.client.tokenizer,
            chunk_size=256,
            chunk_overlap=32)
        # text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        #     tokenizer=self.embed_model.client.tokenizer,
        #     chunk_size=256,
        #     chunk_overlap=32)
        split_docs = text_splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(split_docs,
                                                 self.embed_model
                                                 )
        self.vector_store.save_local(self.index_path)

    def index(self, texts, images):
        total_documents = []
        id_set = set()
        for text in texts:
            file_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
            if file_id in id_set:
                continue
            id_set.add(file_id)
            text_doc = Document(page_content=text,
                                id=file_id,
                                metadata={"source": file_id,
                                          "path": '',
                                          "modality": "text"}
                                )
            total_documents.append(text_doc)
        for image in tqdm.tqdm(images, desc="Indexing images"):
            file_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, image['caption']))
            if file_id in id_set:
                continue
            id_set.add(file_id)
            if self.args.dataset == 'webqa':
                if not os.path.exists(image['path']):
                    print(f'Downloading image from {image["url"]} as {image["path"]}')
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                    }
                    try:
                        save_path = image['path']
                        # Send a GET request to the URL to fetch the image
                        response = requests.get(image['url'], headers=headers)
                        # Check if the request was successful (status code 200)
                        if response.status_code == 200:
                            # Read the image data from the response
                            image_data = BytesIO(response.content)
                            # Open the image using Pillow
                            image = Image.open(image_data).convert("RGB")
                            image.save(save_path)
                    except Exception as e:
                        print(f'Error downloading image from {image["url"]}: {e}')
                        print(f'Skipping.')
                        continue
            image_doc = Document(page_content=image['caption'],
                                 id=file_id,
                                 metadata={"source": file_id,
                                           "modality": "image",
                                           "path": image['path']}
                                 )
            if 'http' in image['path'] or 'https' in image['path']:
                print(image['path'], image['caption'])
                raise NotImplementedError
            '''
            if 'http' in image['path'] or 'https' in image['path']:
                image_url = image['path']
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
                try:
                    # Send a GET request to the URL to fetch the image
                    response = requests.get(image_url, headers=headers)
                    # Check if the request was successful (status code 200)
                    if response.status_code == 200:
                        # Read the image data from the response
                        image_data = BytesIO(response.content)
                        # Open the image using Pillow
                        image = Image.open(image_data).convert("RGB")
                        # Show the image, or you can write custom code to manipulate the image
                        # Close the image
                        image.close()
                    else:
                        continue
                except:
                    continue
            '''
            total_documents.append(image_doc)
        self.index_from_doc(total_documents)

    def load_index(self):
        self.vector_store = FAISS.load_local(self.index_path,
                                             self.embed_model,
                                             allow_dangerous_deserialization=True)

    def query(self, query, top_k=5):
        docs = self.vector_store.similarity_search(query, k=top_k)
        return docs
