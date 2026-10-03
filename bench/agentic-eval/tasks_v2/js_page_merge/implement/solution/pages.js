function mergePages(pages, key) {
 const seen = new Set(), out = [];
 for (const page of pages) for (const item of page) { if (!(key in item)) throw new TypeError('missing key'); const value=item[key]; if (!seen.has(value)) { seen.add(value); out.push(item); } }
 return out.sort((a,b) => a[key] < b[key] ? -1 : a[key] > b[key] ? 1 : 0);
}
module.exports = { mergePages };
