const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),os=require('node:os'),cp=require('node:child_process');
const cfg={"file": "retrying.js", "old": "1-jitter+2*jitter*sample", "new": "1-jitter+jitter*sample", "digest": "16ac423d8b5ae59fc47d77aaaeeafcaad8f249141466d0c4e5f519b7f0f833b6", "test_path": "tests/contract.test.js"}; const root=process.cwd(), target=path.join(root,cfg.file);
const hash=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function run(cwd){const r=cp.spawnSync(process.execPath,['--test',cfg.test_path],{cwd,encoding:'utf8',timeout:8000,env:{...process.env,NO_PROXY:'*',no_proxy:'*'}});return r.status===0;}
if(!fs.existsSync(target)||hash(target)!==cfg.digest)process.exit(2);
if(!fs.existsSync(path.join(root,cfg.test_path)))process.exit(3);
if(!run(root))process.exit(4);
const temp=fs.mkdtempSync(path.join('/dev/shm','v2-grade-'));
try{cp.execFileSync('cp',['-a',root+'/.',temp]);const clone=temp,file=path.join(clone,cfg.file);let text=fs.readFileSync(file,'utf8');if(text.split(cfg.old).length!==2)process.exit(5);fs.writeFileSync(file,text.replace(cfg.old,cfg.new));if(run(clone))process.exit(6);}finally{fs.rmSync(temp,{recursive:true,force:true});}
