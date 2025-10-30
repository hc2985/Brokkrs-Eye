let selectedFile = null;
let downloadUrl = null;

// File input change handler
document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        handleFileSelect(file);
    }
});

// Drag and drop handlers
const uploadBox = document.getElementById('uploadBox');

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f0f2ff';
});

uploadBox.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';

    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
        handleFileSelect(file);
    } else {
        alert('Please drop a CSV file');
    }
});

function handleFileSelect(file) {
    selectedFile = file;

    // Show file info
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('fileInfo').style.display = 'flex';
    document.getElementById('actionSection').style.display = 'block';
    document.getElementById('uploadBox').style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function clearFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('actionSection').style.display = 'none';
    document.getElementById('uploadBox').style.display = 'block';
}

async function processFile() {
    if (!selectedFile) {
        alert('Please select a file first');
        return;
    }

    // Hide upload/action sections, show progress
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('actionSection').style.display = 'none';
    document.getElementById('progressSection').style.display = 'block';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            // Success
            downloadUrl = result.download_url;
            document.getElementById('resultMessage').textContent =
                `Your file "${result.filename}" has been processed successfully!`;
            document.getElementById('downloadBtn').onclick = () => window.location.href = downloadUrl;

            document.getElementById('progressSection').style.display = 'none';
            document.getElementById('resultSection').style.display = 'block';
        } else {
            // Error from server
            throw new Error(result.detail || 'Processing failed');
        }
    } catch (error) {
        // Handle error
        document.getElementById('errorMessage').textContent = error.message;
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('errorSection').style.display = 'block';
    }
}

function reset() {
    // Hide all result/error sections
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';

    // Reset state
    selectedFile = null;
    downloadUrl = null;
    document.getElementById('fileInput').value = '';

    // Show upload box again
    document.getElementById('uploadBox').style.display = 'block';
}
