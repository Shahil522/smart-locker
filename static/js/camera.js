console.log("📷 camera.js loaded");

let video = document.getElementById('camera');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    console.log("✅ Camera started");
  })
  .catch(err => {
    alert("Camera access denied!");
    console.error(err);
  });

// Take snapshot
async function takeSnapshot() {
  console.log("🖼 Taking snapshot...");
  let c = document.createElement('canvas');
  c.width = video.videoWidth || 320;
  c.height = video.videoHeight || 240;
  c.getContext('2d').drawImage(video, 0, 0);

  c.toBlob(async (blob) => {
    console.log("📤 Sending snapshot...");
    let res = await fetch('/api/send_snapshot', {
      method: 'POST',
      body: blob
    });
    let data = await res.json();
    console.log("✅ Snapshot Response:", data);
    alert("Snapshot sent to owner!");
  }, 'image/jpeg');
}

// Request access
async function requestAccess() {
  console.log("🔔 Requesting access...");
  let res = await fetch('/api/request_access', { method: 'POST' });
  let data = await res.json();
  console.log("✅ Access Response:", data);
  alert("Access requested! Please wait for OTP.");
}

// Attach buttons (ensures they exist)
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('btnAccess').onclick = requestAccess;
  document.getElementById('btnSnap').onclick = takeSnapshot;
  console.log("⚙️ Buttons connected");
});
