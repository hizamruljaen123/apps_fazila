// Inisialisasi dashboard setelah DOM siap
document.addEventListener('DOMContentLoaded', function() {

    function initMap() {
        // Inisialisasi peta dengan view default
        const map = L.map('map').setView([5.0621243, 97.3258354], 10);

        // Tambahkan tile layer dari OpenStreetMap
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            maxZoom: 19
        }).addTo(map);

        // Koordinat lokasi (pin default)
        const coordinates = {
            'Baktiya': [5.0621243, 97.3258354],
            'Lhoksukon': [5.0517222, 97.3078233],
            'Langkahan': [4.9211586, 97.1261701],
            'Cot Girek': [4.8616275, 97.2673567],
            'Matangkuli': [5.0306322, 97.2316173],
            'Tanah Luas': [4.9826373, 97.0425453],
            'Stamet Aceh Utara': [5.228798, 96.9449662]
        };

        // Tampilkan marker default pada peta
        const defaultMarkers = [];
        for (const [location, latLng] of Object.entries(coordinates)) {
            const marker = L.marker(latLng).addTo(map)
                .bindPopup(location)
                .openPopup();
            defaultMarkers.push(marker);
        }

        return { map, defaultMarkers };
    }

    const { map, defaultMarkers } = initMap();  // Inisialisasi peta dan marker default

    // Variabel yang berkaitan dengan elemen DOM
    const yearSelect = document.getElementById('yearSelect');
    const yearTabs = document.getElementById('yearTabs');
    const tabContent = document.getElementById('tabContent');
    let floodData = []; // Menyimpan data API secara global

    // Ambil data dari API dan populasi elemen UI
    fetch('/predict')
        .then(response => response.json())
        .then(data => {
            floodData = data; // Simpan data ke variabel global
            populateYearSelect([2023, 2024]);
            // Populasi dropdown tahun
            populateYearTabs([2023, 2024]); 
            // Inisialisasi nav tabs berdasarkan tahun
            displayPredictionChart(data)
        })
        .catch(error => console.error('Error fetching data:', error));

    // Populasi opsi tahun pada select element
    function populateYearSelect(years) {
        years.forEach(year => {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            yearSelect.appendChild(option);
        });
    }

    // Fungsi untuk menampilkan grafik frekuensi kategori prediksi
    function displayPredictionChart(data) {
        // Menghitung frekuensi kategori per tahun
        const frequencyData = {
            2023: { 'Aman': 0, 'Siaga': 0, 'Awas': 0, 'Waspada': 0 },
            2024: { 'Aman': 0, 'Siaga': 0, 'Awas': 0, 'Waspada': 0 }
        };

        data.forEach(item => {
            if (frequencyData[item.Tahun]) {
                frequencyData[item.Tahun][item.Prediksi]++;
            }
        });

        // Siapkan data untuk grafik
        const years = Object.keys(frequencyData);
        const labels = ['Aman', 'Siaga', 'Awas', 'Waspada'];

        const seriesData = labels.map(label => ({
            x: years,
            y: years.map(year => frequencyData[year][label]),
            type: 'bar',
            name: label
        }));

        // Buat grafik menggunakan Plotly
        const layout = {
            title: 'Frekuensi Kategori Prediksi per Tahun',
            barmode: 'group',
            xaxis: { title: 'Tahun' },
            yaxis: { title: 'Frekuensi' }
        };

        Plotly.newPlot('predictionChart', seriesData, layout);
    }

    // Inisialisasi elemen untuk menampilkan grafik
    const chartDiv = document.createElement('div');
    chartDiv.id = 'predictionChart';
    chartDiv.style.width = '100%';
    chartDiv.style.height = '400px';
    document.body.appendChild(chartDiv); // Sesuaikan lokasi penempatan grafik di halaman

    // Event handler untuk tombol "Tampilkan Peta"
    document.getElementById('showMap').onclick = function() {
        const selectedYear = parseInt(yearSelect.value, 10);
        const filteredData = floodData.filter(item => item.Tahun === selectedYear);

        // Hapus marker default terlebih dahulu
        defaultMarkers.forEach(marker => map.removeLayer(marker));

        displayDataOnMap(filteredData);
    };

    // Tampilkan data pada peta berdasarkan tahun yang dipilih
    function displayDataOnMap(data) {
        map.eachLayer(layer => {  // Hapus marker lama
            if (layer.toGeoJSON) map.removeLayer(layer);
        });

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19 }).addTo(map);

        data.forEach(item => {
            const color = item.Prediksi === 'Aman' ? 'green' : item.Prediksi === 'Siaga' ? 'orange' : item.Prediksi === 'Awas' ? 'red' : 'blue';

            L.circleMarker([item.Koordinat.Latitude, item.Koordinat.Longitude], {
                radius: 8,
                fillColor: color,
                color: color,
                weight: 1,
                fillOpacity: 0.8
            }).addTo(map).bindPopup(`<b>${item.Wilayah}</b><br>Prediksi: ${item.Prediksi}<br>Bulan: ${item.Bulan}, Tahun: ${item.Tahun}`);
        });
    }

    // Populasi tabs berdasarkan tahun
    function populateYearTabs(years) {
        years.forEach((year, index) => {
            const tabItem = document.createElement('li');
            tabItem.className = "nav-item";
            tabItem.innerHTML = `
                <a class="nav-link ${index === 0 ? 'active' : ''}" id="tab-${year}" data-toggle="tab" href="#content-${year}" role="tab">${year}</a>
            `;
            yearTabs.appendChild(tabItem);

            const tabContentItem = document.createElement('div');
            tabContentItem.className = `tab-pane fade ${index === 0 ? 'show active' : ''}`;
            tabContentItem.id = `content-${year}`;
            tabContentItem.role = "tabpanel";
            tabContent.appendChild(tabContentItem);

            const filteredData = floodData.filter(item => item.Tahun === year);
            displayDataInTable(filteredData, tabContentItem);
        });
    }

    // Tampilkan data dalam bentuk tabel
    function displayDataInTable(data, container) {
        const table = document.createElement('table');
        table.className = 'table table-bordered';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Wilayah</th>
                    <th>Bulan</th>
                    <th>Prediksi</th>
                    <th>Koordinat (Latitude, Longitude)</th>
                </tr>
            </thead>
        `;

        const tbody = document.createElement('tbody');
        data.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.Wilayah}</td>
                <td>${item.Bulan}</td>
                <td>${item.Prediksi}</td>
                <td>${item.Koordinat.Latitude}, ${item.Koordinat.Longitude}</td>
            `;
            tbody.appendChild(row);
        });

        table.appendChild(tbody);
        container.appendChild(table);
    }

});

function startTraining() {
    // Set the API URL
    const apiUrl = 'http://127.0.0.1:5000/train';

    // Fetch data from the training API
    fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            // Parse the response and display it in the textarea
            const result = `Message: ${data.message}\n` +
                           `Train Accuracy: ${data.train_accuracy}\n` +
                           `Test Accuracy: ${data.test_accuracy}\n` +
                           `Train MSE: ${data.train_mse}\n` +
                           `Test MSE: ${data.test_mse}`;

            // Display the result in the textarea
            document.getElementById('trainResult').value = result;
        })
        .catch(error => {
            // Handle errors
            document.getElementById('trainResult').value = 'Error: ' + error.message;
        });
}

// Array to define the custom order of short month names
const shortMonthOrder = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function loadDataLatih() {
    // Fetch data latih from API
    fetch('/data_train')
        .then(response => response.json())
        .then(data => {
            const tbody = document.querySelector('#dataLatihTable tbody');
            tbody.innerHTML = ''; // Clear previous rows

            // Sort data based on shortMonthOrder and Tahun
            data.sort((a, b) => {
                const monthDiff = shortMonthOrder.indexOf(a.Bulan) - shortMonthOrder.indexOf(b.Bulan);
                if (monthDiff !== 0) {
                    return monthDiff; // Sort by month
                } else {
                    return a.Tahun - b.Tahun; // If months are the same, sort by year
                }
            });

            // Insert sorted data into the table
            data.forEach(item => {
                const row = `
                    <tr>
                        <td>${item.Wilayah}</td>
                        <td>${item.Bulan}</td>
                        <td>${item.Tahun}</td>
                        <td>${item.Curah_Hujan}</td>
                        <td>${item.Suhu}</td>
                        <td>${item.Tinggi_Muka_Air}</td>
                        <td>${item.Potensi_Banjir}</td>
                        <td>
                            <button class="btn btn-danger btn-sm" onclick="deleteDataTrain(${item.id})">Hapus</button>
                        </td>
                    </tr>
                `;
                tbody.insertAdjacentHTML('beforeend', row);
            });
        })
        .catch(error => console.error('Error fetching data latih:', error));
}

function loadDataUji() {
    // Fetch data uji from API
    fetch('/data_test')
        .then(response => response.json())
        .then(data => {
            const tbody = document.querySelector('#dataUjiTable tbody');
            tbody.innerHTML = ''; // Clear previous rows
            console.log(data)
            // Sort data based on shortMonthOrder and Tahun
            data.sort((a, b) => {
                const monthDiff = shortMonthOrder.indexOf(a.Bulan) - shortMonthOrder.indexOf(b.Bulan);
                if (monthDiff !== 0) {
                    return monthDiff; // Sort by month
                } else {
                    return a.Tahun - b.Tahun; // If months are the same, sort by year
                }
            });

            // Insert sorted data into the table
            data.forEach(item => {
                const row = `
                    <tr>
                        <td>${item.Wilayah}</td>
                        <td>${item.Bulan}</td>
                        <td>${item.Tahun}</td>
                        <td>${item.Curah_Hujan}</td>
                        <td>${item.Suhu}</td>
                        <td>${item.Tinggi_Muka_Air}</td>
                        <td>
                            <button class="btn btn-danger btn-sm" onclick="deleteDataTest(${item.id})">Hapus</button>
                        </td>
                    </tr>
                `;
                tbody.insertAdjacentHTML('beforeend', row);
            });
        })
        .catch(error => console.error('Error fetching data uji:', error));
}


// Fungsi untuk menghapus data dari tabel data_banjir (data latih) menggunakan metode GET
async function deleteDataTrain(id) {
    const response = await fetch(`/delete_data_train?id=${id}`, {
        method: 'GET',
    });

    const result = await response.json();
    if (response.ok) {
        console.log(`Data dengan ID ${id} berhasil dihapus:`, result.message);
        // Reload halaman setelah berhasil menghapus data
        loadDataLatih()
    } else {
        console.error(`Gagal menghapus data:`, result.message);
    }
}

// Fungsi untuk menghapus data dari tabel data_uji (data uji) menggunakan metode GET
async function deleteDataTest(id) {
    const response = await fetch(`/delete_data_test?id=${id}`, {
        method: 'GET',
    });

    const result = await response.json();
    if (response.ok) {
        console.log(`Data dengan ID ${id} berhasil dihapus:`, result.message);
        // Reload halaman setelah berhasil menghapus data
        loadDataUji()
    } else {
        console.error(`Gagal menghapus data:`, result.message);
    }
}

document.getElementById('refreshPage').onclick = function() {
    location.reload(); // Ini akan me-refresh halaman
};
