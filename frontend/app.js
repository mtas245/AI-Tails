let stream = null;
let track = null;
let capturedBlob = null;
let zoomFactor = 1;
let hardwareZoom = false;

const screens = [...document.querySelectorAll('.screen')];
const video = document.getElementById('video');
const canvas = document.getElementById('capture-canvas');
const preview = document.getElementById('preview-image');
const fileInput = document.getElementById('file-input');

function showScreen(id){
  screens.forEach(s => s.classList.toggle('active', s.id === id));
  window.scrollTo({top:0,behavior:'smooth'});
}

async function startCamera(){
  try{
    if(stream) stream.getTracks().forEach(t=>t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      video:{
        facingMode:{ideal:'environment'},
        width:{ideal:1920},
        height:{ideal:1440}
      },
      audio:false
    });
    video.srcObject = stream;
    track = stream.getVideoTracks()[0];
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    hardwareZoom = !!caps.zoom;
    document.getElementById('camera-note').textContent = hardwareZoom
      ? 'Hardware-Zoom verfügbar – 1× / 2× / 3× steuern direkt die Smartphone-Kamera.'
      : 'Dieses Gerät meldet keinen Hardware-Zoom. AI-Tails nutzt deshalb einen digitalen Zoom-Fallback.';
  }catch(err){
    document.getElementById('camera-note').textContent =
      'Kamera konnte nicht geöffnet werden. Nutze bitte „Bild hochladen“ oder erlaube den Kamerazugriff im Browser.';
  }
}

async function applyZoom(factor){
  zoomFactor = factor;
  document.querySelectorAll('.zoom-chip').forEach(b=>b.classList.toggle('active', Number(b.dataset.zoom)===factor));
  if(track && hardwareZoom){
    const caps = track.getCapabilities();
    const min = caps.zoom.min ?? 1;
    const max = caps.zoom.max ?? factor;
    const value = Math.min(max, Math.max(min, factor));
    try{
      await track.applyConstraints({advanced:[{zoom:value}]});
      video.style.transform = 'scale(1)';
      return;
    }catch(e){}
  }
  video.style.transform = `scale(${factor})`;
}

function captureFrame(){
  if(!video.videoWidth) return;
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  if(hardwareZoom){
    canvas.width = vw;
    canvas.height = vh;
    canvas.getContext('2d').drawImage(video,0,0,vw,vh);
  }else{
    const cropW = vw / zoomFactor;
    const cropH = vh / zoomFactor;
    const sx = (vw - cropW) / 2;
    const sy = (vh - cropH) / 2;
    canvas.width = Math.round(cropW);
    canvas.height = Math.round(cropH);
    canvas.getContext('2d').drawImage(video,sx,sy,cropW,cropH,0,0,canvas.width,canvas.height);
  }

  canvas.toBlob(blob=>{
    capturedBlob = blob;
    preview.src = URL.createObjectURL(blob);
    showScreen('preview-screen');
  },'image/jpeg',0.94);
}

fileInput.addEventListener('change',()=>{
  const file = fileInput.files[0];
  if(!file) return;
  capturedBlob = file;
  preview.src = URL.createObjectURL(file);
  showScreen('preview-screen');
});

document.getElementById('capture-btn').addEventListener('click',captureFrame);
document.getElementById('retake-btn').addEventListener('click',()=>showScreen('capture-screen'));
document.querySelectorAll('.zoom-chip').forEach(btn=>btn.addEventListener('click',()=>applyZoom(Number(btn.dataset.zoom))));
document.getElementById('new-analysis').addEventListener('click',()=>{
  capturedBlob = null;
  document.getElementById('landmark-panel').classList.remove('open');
  showScreen('capture-screen');
});

document.getElementById('show-landmarks').addEventListener('click',()=>{
  document.getElementById('landmark-panel').classList.add('open');
  document.getElementById('landmark-panel').scrollIntoView({behavior:'smooth',block:'start'});
});
document.getElementById('close-landmarks').addEventListener('click',()=>{
  document.getElementById('landmark-panel').classList.remove('open');
});

function renderResult(data){
  const card = document.getElementById('status-card');
  card.className = 'status-card ' + data.status.level;
  document.getElementById('status-eyebrow').textContent = data.status.eyebrow;
  document.getElementById('status-title').textContent = data.status.title;
  document.getElementById('status-summary').textContent = data.status.summary;
  document.getElementById('face-confidence').textContent = Math.round(data.detector_confidence*100) + '%';
  document.getElementById('quality-label').textContent = data.quality.accepted ? 'Gut' : 'Eingeschränkt';
  document.getElementById('landmark-image').src = data.landmark_image;
  document.getElementById('report-text').textContent = data.status.note;

  const list = document.getElementById('feature-list');
  list.innerHTML = '';
  data.features.forEach(f=>{
    const el = document.createElement('div');
    el.className='feature';
    const score = f.score === null ? 'nicht bewertet' : `${f.score}/2`;
    const sigma = f.deviation_sigma === null ? '' : ` · ${Number(f.deviation_sigma).toFixed(2)}σ`;
    el.innerHTML = `
      <div class="feature-top">
        <span class="feature-label">${f.label}</span>
        <span class="feature-score">${score}${sigma}</span>
      </div>
      <p>${f.text}</p>`;
    list.appendChild(el);
  });
}

document.getElementById('analyze-btn').addEventListener('click',async()=>{
  if(!capturedBlob) return;
  showScreen('loading-screen');

  const copies = [
    'Katzengesicht erkennen …',
    '48 Landmark-Punkte bestimmen …',
    'Gesichtsgeometrie vergleichen …'
  ];
  let i=0;
  const timer=setInterval(()=>{
    i=(i+1)%copies.length;
    document.getElementById('loading-copy').textContent=copies[i];
  },900);

  try{
    const form = new FormData();
    form.append('file',capturedBlob,'capture.jpg');
    const response = await fetch('/analyze',{method:'POST',body:form});
    const data = await response.json();
    if(!response.ok) throw new Error(data.detail || 'Analyse fehlgeschlagen');
    renderResult(data);
    showScreen('result-screen');
  }catch(err){
    alert(err.message);
    showScreen('preview-screen');
  }finally{
    clearInterval(timer);
  }
});

startCamera();
