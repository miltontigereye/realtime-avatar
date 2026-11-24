from pathlib import Path
from collections import Counter
import argparse,hashlib,json,shutil,subprocess
import pikepdf
parser=argparse.ArgumentParser(description='Lossless PDF export with verified text sidecars')
parser.add_argument('source',type=Path)
parser.add_argument('destination',type=Path)
args=parser.parse_args()
SRC=args.source
DEST=args.destination
DEST.mkdir(parents=True,exist_ok=False)
(DEST/'text').mkdir(exist_ok=True)
manifest={'method':'Lossless Flate recompression and PDF object-stream packing. Original retained whenever optimized file is not smaller. Images are not downsampled or re-encoded. Text sidecars are search aids; PDFs retain layout, equations and figures.','papers':[]}
def sha(data):return hashlib.sha256(data).hexdigest()
def streams(pdf):
    result=Counter()
    for obj in pdf.objects:
        if not isinstance(obj,pikepdf.Stream):continue
        if str(obj.get('/Type','')) in {'/ObjStm','/XRef','/Metadata'}:continue
        try:
            data=obj.read_bytes(pikepdf.StreamDecodeLevel.generalized)
            kind='decoded'
        except pikepdf.PdfError:
            data=obj.read_raw_bytes()
            kind='encoded:'+str(obj.get('/Filter',''))
        result[(kind,sha(data))]+=1
    return result
for src in sorted(SRC.iterdir()):
    if not src.is_file() or not src.read_bytes()[:1024].lstrip().startswith(b'%PDF-'):continue
    name=src.name if src.suffix.lower()=='.pdf' else src.name+'.pdf'
    if src.name=='Omni-LiveAvatarpdf':name='Omni-LiveAvatar.pdf'
    dst=DEST/name
    with pikepdf.open(src) as pdf:
        before=streams(pdf)
        pages=len(pdf.pages)
        pdf.save(dst,compress_streams=True,recompress_flate=True,object_stream_mode=pikepdf.ObjectStreamMode.generate,normalize_content=False,fix_metadata_version=False)
    with pikepdf.open(dst) as pdf:
        assert len(pdf.pages)==pages
        stream_equal=streams(pdf)==before
    if not stream_equal or dst.stat().st_size>=src.stat().st_size:
        shutil.copyfile(src,dst)
        method='original_retained_for_exact_stream_preservation' if not stream_equal else 'original_already_smaller'
    else:method='lossless_recompressed'
    original_text=subprocess.check_output(['pdftotext','-layout',str(src),'-'],stderr=subprocess.DEVNULL)
    output_text=subprocess.check_output(['pdftotext','-layout',str(dst),'-'],stderr=subprocess.DEVNULL)
    assert original_text==output_text,src.name+' extracted text changed'
    text=DEST/'text'/(Path(name).stem+'.txt')
    text.write_bytes(output_text)
    item={'source_name':src.name,'pdf':name,'text':str(text.relative_to(DEST)),'pages':pages,'original_bytes':src.stat().st_size,'published_bytes':dst.stat().st_size,'source_sha256':sha(src.read_bytes()),'pdf_sha256':sha(dst.read_bytes()),'text_sha256':sha(output_text),'method':method,'checks':{'page_count_equal':True,'content_and_image_streams_equal':True,'extracted_text_equal':True}}
    manifest['papers'].append(item)
    print(name,pages,'pages',item['original_bytes'],'->',item['published_bytes'],flush=True)
(DEST/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
a=sum(p['original_bytes'] for p in manifest['papers']); b=sum(p['published_bytes'] for p in manifest['papers'])
print('TOTAL',len(manifest['papers']),a,b,'saved',round((1-b/a)*100,2),'%',flush=True)
