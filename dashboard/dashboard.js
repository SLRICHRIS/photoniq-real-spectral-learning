(function () {
  'use strict';
  const data = window.PHOTONIQ_SNAPSHOT;
  if (!data || !data.cross_validation_summary) {
    document.body.textContent = 'Research snapshot unavailable. Check dashboard/data/snapshot.js.';
    return;
  }
  const $ = id => document.getElementById(id);
  const metricRows = data.cross_validation_summary;
  const fullCNN = metricRows.find(x => x.model === 'CNN - 351 wavelengths');
  const sparseCNN = metricRows.find(x => x.model === 'CNN - 64 wavelengths');
  const ridgeFull = metricRows.find(x => x.model === 'Ridge - 351 wavelengths');
  const nf = (value, digits=2) => Number(value).toFixed(digits);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const root = 'https://github.com/SLRICHRIS/photoniq-real-spectral-learning';

  // Stage descriptions mirror actual source code and published run provenance.
  const stages = [
    {name:'Source data', short:'Zenodo measurements', icon:'↧', description:'Collect independent film-thickness labels and corresponding measured optical-density curves.', input:'Zenodo Dataset.zip · 517 films', operation:'Read blade_perovskite_database.csv and curvedata_opticaldensity.csv.', safeguard:'Original archive SHA-256 verified; no synthetic training fallback.', output:'Sample IDs, thickness targets, wavelength-indexed spectral records.'},
    {name:'Dataset audit', short:'Integrity checks', icon:'◈', description:'Validate the specimen-to-spectrum relationship before modeling.', input:'Database labels and 351-channel spectra', operation:'Join by sample ID, check uniqueness, missing values and increasing wavelength axis.', safeguard:'Exclude thickness-derived integrated features; interpolate only within each measured film.', output:'517 usable films, 20 fabrication dates and source provenance.'},
    {name:'Group splitting', short:'Held-out dates', icon:'▦', description:'Create honest experimental evaluation partitions across five outer folds.', input:'517 films with fabrication dates', operation:'Five-fold GroupKFold; independent date-separated validation partition inside each fold.', safeguard:'A fabrication date never appears in train and held-out data simultaneously.', output:'Five held-out test partitions; each film receives exactly one test prediction.'},
    {name:'Channel budgets', short:'16 to 351 wavelengths', icon:'⋮', description:'Study performance under reduced spectral acquisition resolution.', input:'Measured 300–1000 nm optical-density curves', operation:'Uniform channel selections: 16, 32, 64, 128, 256, 351.', safeguard:'Choose channels without using test labels; retrain each CNN independently.', output:'Six distinct model inputs per fold (not active sensing).'},
    {name:'Model training', short:'Ridge and 1D CNN', icon:'⌁', description:'Fit classical regression baselines and compact convolutional models.', input:'Training-only normalised spectral features', operation:'Mean baseline, direct-statistics Ridge, full/subsampled Ridge and per-budget CNN.', safeguard:'CNN early stopping uses only validation dates, never held-out test dates.', output:'Fold checkpoints, validation histories and frozen models.'},
    {name:'Out-of-fold test', short:'Unseen-date inference', icon:'◎', description:'Evaluate frozen models only on films from the outer held-out fabrication dates.', input:'Held-out spectra and their independently measured targets', operation:'Generate one prediction per film; pool out-of-fold scores across 5 folds.', safeguard:'No train/test date overlap; per-fold and pooled metrics recalculated independently.', output:'517 out-of-fold predictions, MAE, RMSE, R² and per-fold scores.'},
    {name:'Error audit', short:'Study limitations', icon:'▤', description:'Examine the distribution, reliability and limitations of the prediction errors.', input:'Saved out-of-fold predictions and fabrication metadata', operation:'Date-cluster bootstrap and stratified error checks by thickness, date, substrate and conditions.', safeguard:'Only descriptive interpretation; external materials and laboratories not validated.', output:'Bias tables, diagnostic figures, uncertainties and reproducibility report.'}
  ];

  function showPage(page) {
    const valid = ['overview','pipeline','results','diagnostics','reproducibility'];
    if (!valid.includes(page)) page = 'overview';
    document.querySelectorAll('.page').forEach(el => {let show = el.id === page; el.classList.toggle('active',show); el.hidden=!show;});
    document.querySelectorAll('.nav-item').forEach(el => {let show = el.dataset.page === page; el.classList.toggle('active',show); if(show)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
    history.replaceState(null,'','#'+page);
    window.scrollTo({top:0,behavior:'instant'});
  }
  document.querySelectorAll('[data-page]').forEach(el=>el.addEventListener('click',()=>showPage(el.dataset.page)));
  document.querySelectorAll('[data-goto]').forEach(el=>el.addEventListener('click',()=>showPage(el.dataset.goto)));
  window.addEventListener('hashchange',()=>showPage(location.hash.slice(1)));

  $('filmCount').textContent=data.run_metadata.valid_joined_films.toLocaleString('en-US');
  $('dateCount').textContent=data.run_metadata.distinct_fabrication_dates;
  $('fullMAE').innerHTML=`${nf(fullCNN.MAE_nm)}<span class="metric-unit"> nm</span>`;
  $('sparseMAE').innerHTML=`${nf(sparseCNN.MAE_nm)}<span class="metric-unit"> nm</span>`;
  $('shaValue').textContent=data.run_metadata.source_sha256;
  $('overviewStrip').innerHTML=stages.map(s=>`<div class="strip-item"><span class="strip-icon" aria-hidden="true">${s.icon}</span><span class="strip-title">${esc(s.name)}</span><span class="strip-caption">${esc(s.short)}</span></div>`).join('');
  $('overviewModelBars').innerHTML=[{name:'Mean baseline',n:metricRows.find(x=>x.model==='Mean baseline').MAE_nm,k:'mean'}, {name:'Ridge · 351',n:ridgeFull.MAE_nm,k:'ridge'}, {name:'CNN · 351',n:fullCNN.MAE_nm,k:'cnn'}, {name:'CNN · 64',n:sparseCNN.MAE_nm,k:'sparse'}].map(x=>`<div class="bar-row"><span class="bar-label">${esc(x.name)}</span><div class="bar-track" role="img" aria-label="${esc(x.name)}: ${nf(x.n)} nanometres mean absolute error"><span class="bar-fill ${x.k}" style="width:${x.n/190*100}%"></span></div><span class="bar-value">${nf(x.n)} nm</span></div>`).join('');

  // Pipeline selection deliberately changes substantive research-method details.
  let selectedStage = 0;
  $('stages').innerHTML=stages.map((s,i)=>`<button type="button" role="tab" class="stage-button" data-stage="${i}" id="stage-tab-${i}" aria-controls="stageDetail" aria-selected="${i===0}"><span class="number">STAGE 0${i+1}</span><span class="symbol">${s.icon}</span><span class="stage-title">${esc(s.name)}</span></button>`).join('');
  function selectStage(i){selectedStage=i;const s=stages[i];document.querySelectorAll('[data-stage]').forEach(x=>x.setAttribute('aria-selected',String(Number(x.dataset.stage)===i)));$('stageDetail').setAttribute('aria-labelledby',`stage-tab-${i}`);$('stageDetail').innerHTML=`<div><span class="detail-eyebrow">STAGE 0${i+1} / 07</span><h3 class="detail-heading">${esc(s.name)}</h3><p class="detail-summary">${esc(s.description)}</p></div><div class="detail-list"><div class="detail-row"><strong>Input</strong><span>${esc(s.input)}</span></div><div class="detail-row"><strong>Process</strong><span>${esc(s.operation)}</span></div><div class="detail-row"><strong>Safeguard</strong><span>${esc(s.safeguard)}</span></div><div class="detail-row"><strong>Output</strong><span>${esc(s.output)}</span></div></div>`;}
  document.querySelectorAll('[data-stage]').forEach(x=>x.addEventListener('click',()=>selectStage(Number(x.dataset.stage))));selectStage(selectedStage);

  let selectedMetric='MAE_nm', selectedMethod='both';
  const budgets=[16,32,64,128,256,351];
  function series(name){return budgets.map(b=>metricRows.find(x=>x.model===`${name} - ${b} wavelengths`));}
  const lineSeries=[{label:'CNN',color:'#416fb5',rows:series('CNN'),key:'cnn'},{label:'Ridge',color:'#9fb4d3',rows:series('Ridge'),key:'ridge'}];
  const labels={MAE_nm:'MAE (nm) · lower is better',RMSE_nm:'RMSE (nm) · lower is better',R2:'R² · higher is better'};
  function drawBudgetChart(){
    const width=960,height=340, left=62,right=35,top=37,bottom=56;
    const visible=lineSeries.filter(s=>selectedMethod==='both'||s.key===selectedMethod);
    const values=visible.flatMap(s=>s.rows.map(r=>Number(r[selectedMetric])));
    const lo=Math.min(...values),hi=Math.max(...values);
    const pad=Math.max((hi-lo)*.18,selectedMetric==='R2'?.045:7);
    const bottomVal=selectedMetric==='R2'?Math.max(-.5,lo-pad):0;
    const topVal=hi+pad;
    const x=i=>left+i*(width-left-right)/5, y=v=>top+(topVal-v)*(height-top-bottom)/(topVal-bottomVal||1);
    const ticks=Array.from({length:5},(_,i)=>bottomVal+i*(topVal-bottomVal)/4);
    let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${labels[selectedMetric]} by wavelength budget for ${selectedMethod}"><text x="${left}" y="18" class="chart-axis-title">${labels[selectedMetric]}</text>`;
    for(const v of ticks){const yy=y(v);svg+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" class="chart-gridline"/><text x="${left-9}" y="${yy+4}" class="chart-tick" text-anchor="end">${selectedMetric==='R2'?v.toFixed(2):v.toFixed(0)}</text>`;}
    budgets.forEach((b,i)=>{svg+=`<text x="${x(i)}" y="${height-25}" class="chart-tick" text-anchor="middle">${b}</text>`;});
    svg+=`<text x="${(left+width-right)/2}" y="${height-3}" text-anchor="middle" class="chart-axis-title">Measured wavelength channels per model</text>`;
    visible.forEach(s=>{
      const points=s.rows.map((row,i)=>`${x(i)},${y(Number(row[selectedMetric]))}`).join(' ');
      svg+=`<polyline fill="none" stroke="${s.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${points}"/>`;
      s.rows.forEach((row,i)=>{const v=Number(row[selectedMetric]);svg+=`<circle class="chart-point" cx="${x(i)}" cy="${y(v)}" r="5" fill="${s.color}" stroke="white" stroke-width="2"><title>${s.label}: ${budgets[i]} channels · ${labels[selectedMetric]} ${nf(v,selectedMetric==='R2'?3:2)}</title></circle>`;});
    });svg+='</svg>';
    $('budgetChart').innerHTML=svg;
    document.querySelectorAll('[data-metric]').forEach(x=>x.classList.toggle('selected',x.dataset.metric===selectedMetric));
  }
  document.querySelectorAll('[data-metric]').forEach(x=>x.addEventListener('click',()=>{selectedMetric=x.dataset.metric;drawBudgetChart();}));
  $('methodFilter').addEventListener('change',e=>{selectedMethod=e.target.value;drawBudgetChart();});
  drawBudgetChart();

  const order=['Mean baseline','Spectral-statistics Ridge',...budgets.flatMap(b=>[`Ridge - ${b} wavelengths`,`CNN - ${b} wavelengths`])];
  $('resultsTable').innerHTML=order.map(m=>{const x=metricRows.find(d=>d.model===m);const best=m==='CNN - 64 wavelengths';return `<tr data-model="${esc(m)}" ${best?'class="row-highlight"':''}><td>${esc(m)}</td><td>${x.wavelength_budget||'—'}</td><td class="${best?'best-cell':''}">${nf(x.MAE_nm)}</td><td>${nf(x.RMSE_nm)}</td><td>${nf(x.R2,3)}</td></tr>`;}).join('');
  $('resultsTable').addEventListener('click',e=>{const row=e.target.closest('[data-model]');if(!row)return;document.querySelectorAll('#resultsTable tr').forEach(t=>t.classList.toggle('row-highlight',t===row));const m=row.dataset.model;if(m.startsWith('CNN -')){$('methodFilter').value='cnn';selectedMethod='cnn';drawBudgetChart();}else if(m.startsWith('Ridge -')){$('methodFilter').value='ridge';selectedMethod='ridge';drawBudgetChart();}});

  function drawDiagnostics(){
    const group=$('diagnosticFilter').value;
    const rows=data.error_diagnostics.filter(x=>x.dimension===group).sort((a,b)=>group==='fabrication_date'?b.CNN_MAE_nm-a.CNN_MAE_nm:0);
    $('diagnosticTable').innerHTML=rows.map(r=>`<tr><td>${esc(r.group)}</td><td>${r.n_films}</td><td>${nf(r.CNN_MAE_nm)}</td><td>${Number(r.CNN_bias_nm)>0?'+':''}${nf(r.CNN_bias_nm)}</td><td>${nf(r.Ridge_MAE_nm)}</td></tr>`).join('');
  }
  $('diagnosticFilter').addEventListener('change',drawDiagnostics);drawDiagnostics();
  $('copyHash').addEventListener('click',()=>{
    const hash=data.run_metadata.source_sha256;
    if(navigator.clipboard && navigator.clipboard.writeText){navigator.clipboard.writeText(hash).then(()=>{$('copyHash').textContent='Copied';},()=>{$('copyHash').textContent='Select checksum';});}
    else{$('copyHash').textContent='Select checksum';const s=window.getSelection(),r=document.createRange();r.selectNodeContents($('shaValue'));s.removeAllRanges();s.addRange(r);}
  });
  showPage(location.hash.replace('#','') || 'overview');
})();
