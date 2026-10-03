function delayFor(attempt,base,cap,jitter=0,sample=.5){ if(!Number.isInteger(attempt)||attempt<1||![base,cap,jitter,sample].every(Number.isFinite)||base<0||cap<base||jitter<0||jitter>1||sample<0||sample>1)throw new RangeError('invalid range'); const raw=Math.min(cap,base*2**(attempt-1));return raw*(1-jitter+jitter*sample); }
module.exports={delayFor};
