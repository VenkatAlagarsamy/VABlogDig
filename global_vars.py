#//huggingface.co/settings/tokens using va@gmail
HUGGINGFACEHUB_API_TOKEN='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'

#using v.a@gmail.com
OPENAI_API_KEY='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'

#using github 
QDRANT_API_KEY='XBO7p356R4TQ4DGkgkI48Nx2rbUXlWTP8NOdx8hfAq_OkOvnT9qe1Q'
QDRANT_END_POINT='https://27994267-5aae-4840-b101-85dba72dfcd8.us-east4-0.gcp.cloud.qdrant.io:6333'

#Qdrant Collection Name
QDRANT_COLLECTION_NAME='venkat-blog-collection'

def get_openai_api_key() :
    return OPENAI_API_KEY

def get_hf_api_key() :
    return HUGGINGFACEHUB_API_TOKEN

def get_qdrant_api_key() :
    return QDRANT_API_KEY

def get_qdrant_url() :
    return QDRANT_END_POINT

def get_qdrant_collection_name() :
    return QDRANT_COLLECTION_NAME