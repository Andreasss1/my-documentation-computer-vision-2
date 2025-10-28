<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard YOLO Object Detection</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .subtitle {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .dashboard {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        .video-container {
            position: relative;
            background: #000;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        #videoFeed {
            width: 100%;
            height: auto;
            display: block;
        }

        .status-panel {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .status-card {
            flex: 1;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .status-main {
            flex: 2;
        }

        .status-card h2 {
            font-size: 1.2em;
            margin-bottom: 15px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .status-value {
            font-size: 3em;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        .status-pass {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }

        .status-ng {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
            color: white;
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.02);
            }
        }

        .info-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .counter-value {
            font-size: 2.5em;
            font-weight: bold;
        }

        .timer-value {
            font-size: 2em;
            font-weight: bold;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .info-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        .info-item strong {
            display: block;
            color: #667eea;
            margin-bottom: 5px;
            font-size: 0.9em;
        }

        .info-item span {
            font-size: 1.1em;
            color: #333;
        }

        footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.8;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .status-panel {
                flex-direction: column;
            }
            
            h1 {
                font-size: 1.8em;
            }
            
            .status-value {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 Dashboard Deteksi Objek YOLO</h1>
            <p class="subtitle">Monitoring Real-Time Gelembung Standar</p>
        </header>

        <div class="dashboard">
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="Video Feed" onerror="handleImageError()">
            </div>

            <div class="status-panel">
                <div class="status-card status-main" id="statusCard">
                    <h2>Status Inspeksi</h2>
                    <div class="status-value" id="statusValue">PASS</div>
                </div>

                <div class="status-card info-card">
                    <h2>Counter Deteksi</h2>
                    <div class="counter-value" id="counterValue">0</div>
                    <small>/ 4 untuk NG</small>
                </div>

                <div class="status-card info-card" id="timerCard" style="display: none;">
                    <h2>Waktu Tersisa NG</h2>
                    <div class="timer-value" id="timerValue">0.0s</div>
                </div>
            </div>

            <div class="info-grid">
                <div class="info-item">
                    <strong>Model</strong>
                    <span>new-sg-model.pt</span>
                </div>
                <div class="info-item">
                    <strong>Confidence</strong>
                    <span>> 0.4</span>
                </div>
                <div class="info-item">
                    <strong>Target Objek</strong>
                    <span>Gelembung Standar</span>
                </div>
                <div class="info-item">
                    <strong>Durasi NG</strong>
                    <span>5 detik</span>
                </div>
            </div>
        </div>

        <footer>
            <p>© 2024 YOLO Object Detection System</p>
        </footer>
    </div>

    <script>
        const statusCard = document.getElementById('statusCard');
        const statusValue = document.getElementById('statusValue');
        const counterValue = document.getElementById('counterValue');
        const timerCard = document.getElementById('timerCard');
        const timerValue = document.getElementById('timerValue');

        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    // Update status
                    statusValue.textContent = data.status;
                    
                    // Update card styling
                    statusCard.className = 'status-card status-main';
                    if (data.status === 'NG') {
                        statusCard.classList.add('status-ng');
                    } else {
                        statusCard.classList.add('status-pass');
                    }
                    
                    // Update counter
                    counterValue.textContent = data.counter;
                    
                    // Update timer
                    if (data.remaining_time > 0) {
                        timerCard.style.display = 'block';
                        timerValue.textContent = data.remaining_time.toFixed(1) + 's';
                    } else {
                        timerCard.style.display = 'none';
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }

        function handleImageError() {
            const videoFeed = document.getElementById('videoFeed');
            videoFeed.style.background = '#000';
            videoFeed.alt = 'Menunggu koneksi kamera...';
        }

        // Update status setiap 100ms untuk responsif
        setInterval(updateStatus, 100);
        
        // Initial update
        updateStatus();
    </script>
</body>
</html>
