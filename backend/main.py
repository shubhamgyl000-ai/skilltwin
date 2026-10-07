from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pathlib import Path
import re,io,json
BASE=Path(__file__).resolve().parent.parent
ROLES=json.loads((BASE/'data/roles.json').read_text())
SKILLS=json.loads((BASE/'data/skills.json').read_text())
app=FastAPI(title='SkillTwin AI',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
def get_text(data,name):
    if name.lower().endswith('.pdf'): return '\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(data)).pages)
    return data.decode('utf-8','ignore')
def extract_skills(text):
    low=text.lower(); found=[]
    for skill,aliases in SKILLS.items():
        if any(re.search(r'(?<!\\w)'+re.escape(a.lower())+r'(?!\\w)',low) for a in [skill,*aliases]): found.append(skill)
    return sorted(set(found))
def match(skills,role):
    r=next((x for x in ROLES if x['name'].lower()==role.lower()),None)
    if not r:return None
    req=set(r['skills']); have=set(skills); matched=sorted(req&have); missing=sorted(req-have)
    return {'role':r['name'],'score':round(100*len(matched)/len(req)),'matched':matched,'missing':missing,'roadmap':missing[:5]}
@app.get('/api/health')
def health(): return {'status':'ok','service':'SkillTwin AI'}
@app.get('/api/roles')
def roles(): return ROLES
@app.post('/api/analyze')
async def analyze(file:UploadFile=File(...)):
    data=await file.read(); text=get_text(data,file.filename or 'resume.txt'); skills=extract_skills(text)
    matches=sorted([match(skills,r['name']) for r in ROLES],key=lambda x:x['score'],reverse=True)
    return {'filename':file.filename,'skills':skills,'top_roles':matches,'text_preview':text[:1500]}
@app.post('/api/what-if')
async def what_if(payload:dict):
    skills=set(payload.get('skills',[])); add=set(payload.get('add_skills',[])); role=payload.get('role','')
    return {'before':match(sorted(skills),role),'after':match(sorted(skills|add),role),'added':sorted(add)}
