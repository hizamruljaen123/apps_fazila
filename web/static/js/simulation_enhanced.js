// Enhanced simulation.js file for Flood Prediction Simulation
document.addEventListener('DOMContentLoaded', function() {
    // Helper function to format numbers
    function formatNumber(num, decimals = 4) {
        return num.toFixed(decimals);
    }
    
    // Helper function to format percentages
    function formatPercent(num, decimals = 2) {
        return (num * 100).toFixed(decimals) + '%';
    }
    
    // LM Simulation
    document.getElementById('lm-run').addEventListener('click', function() {
        const hiddenSize = parseInt(document.getElementById('lm-hidden-size').value);
        const iterations = parseInt(document.getElementById('lm-iterations').value);
        const learningRate = parseFloat(document.getElementById('lm-learning-rate').value);
        
        // Disable button and show loading
        const button = this;
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Simulasi sedang berjalan...';
        
        // Call API
        fetch('/simulate_lm', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                hidden_size: hiddenSize,
                iterations: iterations,
                learning_rate: learningRate
            }),
        })
        .then(response => response.json())
        .then(data => {
            // Process response and update UI
            updateLMResults(data, hiddenSize, iterations);
            
            // Reset button
            button.disabled = false;
            button.innerHTML = 'Jalankan Simulasi LM';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Terjadi kesalahan: ' + error);
            
            // Reset button
            button.disabled = false;
            button.innerHTML = 'Jalankan Simulasi LM';
        });
    });
    
    // RNN Simulation
    document.getElementById('rnn-run').addEventListener('click', function() {
        const hiddenSize = parseInt(document.getElementById('rnn-hidden-size').value);
        const epochs = parseInt(document.getElementById('rnn-epochs').value);
        const learningRate = parseFloat(document.getElementById('rnn-learning-rate').value);
        const batchSize = parseInt(document.getElementById('rnn-batch-size').value);
        
        // Disable button and show loading
        const button = this;
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Simulasi sedang berjalan...';
        
        // Call API
        fetch('/simulate_rnn', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                hidden_size: hiddenSize,
                epochs: epochs,
                learning_rate: learningRate,
                batch_size: batchSize
            }),
        })
        .then(response => response.json())
        .then(data => {
            // Process response and update UI
            updateRNNResults(data, hiddenSize, epochs, batchSize);
            
            // Reset button
            button.disabled = false;
            button.innerHTML = 'Jalankan Simulasi RNN';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Terjadi kesalahan: ' + error);
            
            // Reset button
            button.disabled = false;
            button.innerHTML = 'Jalankan Simulasi RNN';
        });
    });
    
    // Function to update LM results in UI
    function updateLMResults(data, hiddenSize, iterations) {
        // Navigate to data preparation step
        document.querySelector('#lmStepTabs a[href="#lm-step2"]').click();
        
        // Update data info with enhanced database statistics
        let dataStatsHtml = `
            <h5>Informasi Data</h5>
            <table class="table table-bordered">
                <tr><th>Fitur</th><td>Curah Hujan, Suhu, Tinggi Muka Air</td></tr>
                <tr><th>Target</th><td>Potensi Banjir (Aman, Waspada, Siaga, Awas)</td></tr>
                <tr><th>Training Set</th><td>70%</td></tr>
                <tr><th>Testing Set</th><td>30%</td></tr>
            </table>
            
            <h5 class="mt-3">Statistik Fitur</h5>
            <table class="table table-bordered">
                <thead>
                    <tr>
                        <th>Fitur</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Rata-rata</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        if (data.data_stats && data.data_stats.feature_stats) {
            const stats = data.data_stats.feature_stats;
            dataStatsHtml += `
                <tr>
                    <td>Curah Hujan</td>
                    <td>${formatNumber(stats.curah_hujan.min, 1)}</td>
                    <td>${formatNumber(stats.curah_hujan.max, 1)}</td>
                    <td>${formatNumber(stats.curah_hujan.avg, 1)}</td>
                </tr>
                <tr>
                    <td>Suhu</td>
                    <td>${formatNumber(stats.suhu.min, 1)}</td>
                    <td>${formatNumber(stats.suhu.max, 1)}</td>
                    <td>${formatNumber(stats.suhu.avg, 1)}</td>
                </tr>
                <tr>
                    <td>Tinggi Muka Air</td>
                    <td>${formatNumber(stats.tinggi_air.min, 1)}</td>
                    <td>${formatNumber(stats.tinggi_air.max, 1)}</td>
                    <td>${formatNumber(stats.tinggi_air.avg, 1)}</td>
                </tr>
            `;
        }
        
        dataStatsHtml += `
                </tbody>
            </table>
        `;
        
        document.getElementById('lm-data-info').innerHTML = dataStatsHtml;
        
        // After a delay, move to next step
        setTimeout(() => {
            // Navigate to model training step
            document.querySelector('#lmStepTabs a[href="#lm-step3"]').click();
            
            // Update training progress with detailed iteration information
            let trainingHtml = `
                <h5>Progress Pelatihan</h5>
                <div class="progress mb-3">
                    <div class="progress-bar bg-success" role="progressbar" style="width: 100%" 
                         aria-valuenow="100" aria-valuemin="0" aria-valuemax="100">100%</div>
                </div>
                <p>Algoritma Levenberg-Marquardt telah selesai dijalankan dengan ${iterations} iterasi.</p>
                <table class="table table-bordered">
                    <thead>
                        <tr>
                            <th>Metrik</th>
                            <th>Nilai</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Train MSE</td>
                            <td>${formatNumber(data.metrics.train_mse)}</td>
                        </tr>
                        <tr>
                            <td>Test MSE</td>
                            <td>${formatNumber(data.metrics.test_mse)}</td>
                        </tr>
                    </tbody>
                </table>
            `;
            
            // Add iteration data table if available
            if (data.iteration_data && data.iteration_data.length > 0) {
                trainingHtml += `
                    <h5 class="mt-4">Detail Iterasi</h5>
                    <div class="table-responsive">
                        <table class="table table-bordered table-sm">
                            <thead>
                                <tr>
                                    <th>Iterasi</th>
                                    <th>Train Loss</th>
                                    <th>Test Loss</th>
                                    <th>Train Acc</th>
                                    <th>Test Acc</th>
                                </tr>
                            </thead>
                            <tbody>
                `;
                
                // Show subset of iterations if there are many
                const showMaxRows = 10;
                const iterationData = data.iteration_data;
                const skipFactor = Math.max(1, Math.floor(iterationData.length / showMaxRows));
                
                for (let i = 0; i < iterationData.length; i += skipFactor) {
                    const iter = iterationData[i];
                    trainingHtml += `
                        <tr>
                            <td>${iter.iteration}</td>
                            <td>${formatNumber(iter.train_loss)}</td>
                            <td>${formatNumber(iter.test_loss)}</td>
                            <td>${formatPercent(iter.train_acc)}</td>
                            <td>${formatPercent(iter.test_acc)}</td>
                        </tr>
                    `;
                }
                
                // Ensure the last row is always shown
                if ((iterationData.length - 1) % skipFactor !== 0) {
                    const lastIter = iterationData[iterationData.length - 1];
                    trainingHtml += `
                        <tr>
                            <td>${lastIter.iteration}</td>
                            <td>${formatNumber(lastIter.train_loss)}</td>
                            <td>${formatNumber(lastIter.test_loss)}</td>
                            <td>${formatPercent(lastIter.train_acc)}</td>
                            <td>${formatPercent(lastIter.test_acc)}</td>
                        </tr>
                    `;
                }
                
                trainingHtml += `
                            </tbody>
                        </table>
                    </div>
                `;
            }
            
            document.getElementById('lm-training-progress').innerHTML = trainingHtml;
            
            // After a delay, show final results
            setTimeout(() => {
                // Navigate to results step
                document.querySelector('#lmStepTabs a[href="#lm-step4"]').tab('show');
                
                // Update metrics and set configuration display
                document.getElementById('lm-hidden-size-display').textContent = hiddenSize;
                document.getElementById('lm-iterations-display').textContent = iterations;
                
                // Update metrics
                document.getElementById('lm-metrics').innerHTML = `
                    <h5>Metrik Evaluasi</h5>
                    <table class="table table-bordered">
                        <thead>
                            <tr>
                                <th>Metrik</th>
                                <th>Training</th>
                                <th>Testing</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Mean Squared Error</td>
                                <td>${formatNumber(data.metrics.train_mse)}</td>
                                <td>${formatNumber(data.metrics.test_mse)}</td>
                            </tr>
                            <tr>
                                <td>Accuracy</td>
                                <td>${formatPercent(data.metrics.train_acc)}</td>
                                <td>${formatPercent(data.metrics.test_acc)}</td>
                            </tr>
                        </tbody>
                    </table>
                `;
                
                // Update performance summary
                let perfSummary = '';
                if (data.prediction_distribution) {
                    perfSummary = `
                        <h5>Distribusi Prediksi</h5>
                        <table class="table table-bordered table-sm">
                            <thead>
                                <tr>
                                    <th>Potensi Banjir</th>
                                    <th>Jumlah Prediksi</th>
                                </tr>
                            </thead>
                            <tbody>
                    `;
                    
                    for (const [label, count] of Object.entries(data.prediction_distribution)) {
                        perfSummary += `
                            <tr>
                                <td>${label}</td>
                                <td>${count}</td>
                            </tr>
                        `;
                    }
                    
                    perfSummary += `
                            </tbody>
                        </table>
                        <p class="mt-3">
                            <strong>Akurasi:</strong> ${formatPercent(data.metrics.test_acc)}<br>
                            <strong>Mean Squared Error:</strong> ${formatNumber(data.metrics.test_mse)}
                        </p>
                    `;
                }
                
                document.getElementById('lm-performance-summary').innerHTML = perfSummary || 'Data performa belum tersedia';
                
                // Update all visualizations
                document.getElementById('lm-loss-plot').innerHTML = 
                    `<img src="data:image/png;base64,${data.visualizations.loss_plot}" alt="Loss Plot" class="img-fluid">`;
                document.getElementById('lm-tsne-plot').innerHTML = 
                    `<img src="data:image/png;base64,${data.visualizations.tsne_plot}" alt="t-SNE Plot" class="img-fluid">`;
                document.getElementById('lm-cm-plot').innerHTML = 
                    `<img src="data:image/png;base64,${data.visualizations.confusion_matrix}" alt="Confusion Matrix" class="img-fluid">`;
                document.getElementById('lm-distribution-plot').innerHTML = 
                    `<img src="data:image/png;base64,${data.visualizations.distribution_plot}" alt="Distribution Plot" class="img-fluid">`;
                
                // Add feature importance if available
                if (data.visualizations.feature_importance) {
                    document.getElementById('lm-feature-importance').innerHTML = 
                        `<img src="data:image/png;base64,${data.visualizations.feature_importance}" alt="Feature Importance" class="img-fluid">`;
                }
                
                // Add correlation matrix if available
                if (data.visualizations.correlation_matrix) {
                    document.getElementById('lm-correlation-matrix').innerHTML = 
                        `<img src="data:image/png;base64,${data.visualizations.correlation_matrix}" alt="Correlation Matrix" class="img-fluid">`;
                }
                
                document.getElementById('lm-map-container').innerHTML = data.visualizations.map_html;
            }, 1500);
        }, 1500);
    }
    
    // Function to update RNN results in UI
    function updateRNNResults(data, hiddenSize, epochs, batchSize) {
        // Navigate to data preparation step
        document.querySelector('#rnnStepTabs a[href="#rnn-step2"]').click();
        
        // Update data info with enhanced database statistics
        let dataStatsHtml = `
            <h5>Informasi Data</h5>
            <table class="table table-bordered">
                <tr><th>Fitur</th><td>Curah Hujan, Suhu, Tinggi Muka Air</td></tr>
                <tr><th>Target</th><td>Potensi Banjir (Aman, Waspada, Siaga, Awas)</td></tr>
                <tr><th>Training Set</th><td>70%</td></tr>
                <tr><th>Testing Set</th><td>30%</td></tr>
                <tr><th>Format Input</th><td>Tensor 3D (batch, sequence_length, features)</td></tr>
            </table>
            
            <h5 class="mt-3">Statistik Fitur</h5>
            <table class="table table-bordered">
                <thead>
                    <tr>
                        <th>Fitur</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Rata-rata</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        if (data.data_stats && data.data_stats.feature_stats) {
            const stats = data.data_stats.feature_stats;
            dataStatsHtml += `
                <tr>
                    <td>Curah Hujan</td>
                    <td>${formatNumber(stats.curah_hujan.min, 1)}</td>
                    <td>${formatNumber(stats.curah_hujan.max, 1)}</td>
                    <td>${formatNumber(stats.curah_hujan.avg, 1)}</td>
                </tr>
                <tr>
                    <td>Suhu</td>
                    <td>${formatNumber(stats.suhu.min, 1)}</td>
                    <td>${formatNumber(stats.suhu.max, 1)}</td>
                    <td>${formatNumber(stats.suhu.avg, 1)}</td>
                </tr>
                <tr>
                    <td>Tinggi Muka Air</td>
                    <td>${formatNumber(stats.tinggi_air.min, 1)}</td>
                    <td>${formatNumber(stats.tinggi_air.max, 1)}</td>
                    <td>${formatNumber(stats.tinggi_air.avg, 1)}</td>
                </tr>
            `;
        }
        
        dataStatsHtml += `
                </tbody>
            </table>
            
            <h5 class="mt-3">Distribusi Label</h5>
            <div class="row">
                <div class="col-md-6">
                    <table class="table table-bordered table-sm">
                        <thead>
                            <tr>
                                <th>Potensi Banjir</th>
                                <th>Jumlah Data</th>
                            </tr>
                        </thead>
                        <tbody>
        `;
        
        if (data.data_stats && data.data_stats.label_counts) {
            Object.entries(data.data_stats.label_counts).forEach(([label, count]) => {
                dataStatsHtml += `
                    <tr>
                        <td>${label}</td>
                        <td>${count}</td>
                    </tr>
                `;
            });
        }
        
        dataStatsHtml += `
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        document.getElementById('rnn-data-info').innerHTML = dataStatsHtml;
        
        // After a delay, move to next step
        setTimeout(() => {
            // Navigate to model training step
            document.querySelector('#rnnStepTabs a[href="#rnn-step3"]').click();
            
            // Update training progress
            document.getElementById('rnn-training-progress').innerHTML = `
                <h5>Progress Pelatihan</h5>
                <div class="progress mb-3">
                    <div class="progress-bar bg-success" role="progressbar" style="width: 100%" 
                         aria-valuenow="100" aria-valuemin="0" aria-valuemax="100">100%</div>
                </div>
                <p>Model RNN telah selesai dilatih dengan ${epochs} epochs.</p>
                <table class="table table-bordered">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Nilai</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Hidden Size</td>
                            <td>${hiddenSize}</td>
                        </tr>
                        <tr>
                            <td>Learning Rate</td>
                            <td>${formatNumber(data.metrics.train_mse)}</td>
                        </tr>
                        <tr>
                            <td>Batch Size</td>
                            <td>${batchSize}</td>
                        </tr>
                    </tbody>
                </table>
                <p class="mt-3">Loss akhir training: ${formatNumber(data.metrics.train_mse)}<br>
                Loss akhir testing: ${formatNumber(data.metrics.test_mse)}</p>
            `;
            
            // Create epoch table
            let epochTableHTML = `
                <h5>Tabel Progress Epoch</h5>
                <div class="table-responsive">
                    <table class="table table-bordered table-striped">
                        <thead>
                            <tr>
                                <th>Epoch</th>
                                <th>Train Loss</th>
                                <th>Test Loss</th>
                                <th>Train Accuracy</th>
                                <th>Test Accuracy</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            // Only show a subset of epochs if there are many
            const epochs = data.epoch_data;
            const showAllEpochs = epochs.length <= 10;
            const epochsToShow = showAllEpochs ? epochs : [
                epochs[0],
                ...epochs.filter((_, i) => i > 0 && i < epochs.length - 1 && i % Math.ceil(epochs.length / 5) === 0),
                epochs[epochs.length - 1]
            ];
            
            epochsToShow.forEach(epoch => {
                epochTableHTML += `
                    <tr>
                        <td>${epoch.epoch}</td>
                        <td>${formatNumber(epoch.train_loss)}</td>
                        <td>${formatNumber(epoch.test_loss)}</td>
                        <td>${formatPercent(epoch.train_acc)}</td>
                        <td>${formatPercent(epoch.test_acc)}</td>
                    </tr>
                `;
            });
            
            epochTableHTML += `
                        </tbody>
                    </table>
                </div>
                ${!showAllEpochs ? `<p class="text-muted small">Showing ${epochsToShow.length} of ${epochs.length} epochs</p>` : ''}
            `;
            
            document.getElementById('rnn-epoch-table').innerHTML = epochTableHTML;
            
            // After a delay, show final results
            setTimeout(() => {
                // Navigate to results step
                document.querySelector('#rnnStepTabs a[href="#rnn-step4"]').tab('show');
                
                // Update metrics
                document.getElementById('rnn-metrics').innerHTML = `
                    <h5>Metrik Evaluasi</h5>
                    <table class="table table-bordered">
                        <thead>
                            <tr>
                                <th>Metrik</th>
                                <th>Training</th>
                                <th>Testing</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Mean Squared Error</td>
                                <td>${formatNumber(data.metrics.train_mse)}</td>
                                <td>${formatNumber(data.metrics.test_mse)}</td>
                            </tr>
                            <tr>
                                <td>Accuracy</td>
                                <td>${formatPercent(data.metrics.train_acc)}</td>
                                <td>${formatPercent(data.metrics.test_acc)}</td>
                            </tr>
                        </tbody>
                    </table>
                `;
                
                // Update visualizations
                document.getElementById('rnn-loss-plot').innerHTML = 
                    `<img src="data:image/png;base64,${data.visualizations.loss_plot}" alt="Loss Plot" class="img-fluid">`;
                document.getElementById('rnn-tsne-plot').innerHTML = 
                    `<img src="data:image/png;base64,${data.visualizations.tsne_plot}" alt="t-SNE Plot" class="img-fluid">`;
                document.getElementById('rnn-cm-plot').innerHTML = 
                    `<img src="data:image/png;base64,${data.visualizations.confusion_matrix}" alt="Confusion Matrix" class="img-fluid">`;
                document.getElementById('rnn-dist-plot').innerHTML = 
                    `<img src="data:image/png;base64,${data.visualizations.distribution_plot}" alt="Distribution Plot" class="img-fluid">`;
                document.getElementById('rnn-map-container').innerHTML = data.visualizations.map_html;
            }, 1500);
        }, 1500);
    }
});
