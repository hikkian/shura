const {createHash}=require('node:crypto');
function enabled(flag,userId,percentage,salt=''){if(typeof flag!=='string'||!flag||typeof userId!=='string'||!userId||!Number.isFinite(percentage)||percentage<0||percentage>100)throw new RangeError('invalid input');const d=createHash('sha256').update(`${salt}\0${flag}\0${userId}`,'utf8').digest();let n=0n;for(const b of d.subarray(0,8))n=(n<<8n)|BigInt(b);return Number(n%10000n)<percentage*100;}
module.exports={enabled};
