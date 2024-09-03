// Inisialisasi dashboard setelah DOM siap
document.addEventListener('DOMContentLoaded', function() {
    // Initialize map
    const map = L.map('map').setView([5.0517222, 97.3078233], 9);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
    }).addTo(map);

    // Initialize year and month select elements
    const yearSelect = document.getElementById('yearSelect');
    const monthSelect = document.getElementById('monthSelect');

    // Populate year and month options
    populateYearMonthSelect();

    // Attach event listener to navigation button
    document.getElementById('navigateButton').onclick = function() {
        const selectedYear = yearSelect.value;
        const selectedMonth = monthSelect.value;
        fetchDataAndDisplay(selectedYear, selectedMonth);
    };

    // Fetch data and display in table and map
    function fetchDataAndDisplay(year, month) {
        const apiUrl = `http://127.0.0.1:5000/predict?year=${year}&month=${month}`;
        fetch(apiUrl)
            .then(response => response.json())
            .then(data => {
                displayDataInTable(data);
                displayDataOnMap(data, map);
            })
            .catch(error => console.error('Error fetching data:', error));
    }

    // Populate year and month select options
    function populateYearMonthSelect() {
        const years = [2023, 2024, 2025];
        const months = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"];

        years.forEach(year => {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            yearSelect.appendChild(option);
        });

        months.forEach((month, index) => {
            const option = document.createElement('option');
            option.value = month;
            option.textContent = month;
            monthSelect.appendChild(option);
        });
    }

    // Display data in table
    function displayDataInTable(data) {
        const tableBody = document.getElementById('tableBody');
        tableBody.innerHTML = ''; // Clear existing table rows

        data.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${item.Wilayah}</td>
                <td>${item.Bulan}</td>
                <td>${item.Tahun}</td>
                <td>${item.Prediksi}</td>
            `;
            tableBody.appendChild(row);
        });
    }

    // Display data on map
    function displayDataOnMap(data, map) {
        // Clear existing markers
        map.eachLayer((layer) => {
            if (!!layer.toGeoJSON) {
                map.removeLayer(layer);
            }
        });

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
        }).addTo(map);

        data.forEach(item => {
            let color;
            switch(item.Prediksi) {
                case 'Aman':
                    color = 'green';
                    break;
                case 'Siaga':
                    color = 'orange';
                    break;
                case 'Awas':
                    color = 'red';
                    break;
                default:
                    color = 'blue';
            }

            L.circleMarker([item.Koordinat.Latitude, item.Koordinat.Longitude], {
                radius: 8,
                fillColor: color,
                color: color,
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(map).bindPopup(`<b>${item.Wilayah}</b><br>${item.Prediksi}`);
        });
    }
});

// Fungsi utama untuk menampilkan grafik
// Fungsi utama untuk menampilkan grafik dan tabel
function displayFloodPredictionChartsAndTable() {
// Fetch data from the predict endpoint
fetch('http://127.0.0.1:5000/predict')
    .then(response => response.json())
    .then(data => {
        // Group data by year
        var groupedData = groupDataByYear(data);

        // Get the container for the charts
        var chartsContainer = document.getElementById('charts');
        chartsContainer.innerHTML = ''; // Clear existing charts

        // Get the container for the tables
        var tablesContainer = document.getElementById('tables');
        tablesContainer.innerHTML = ''; // Clear existing tables

        // Loop through each year and create a chart and table
        Object.keys(groupedData).forEach(year => {
            // Create a new div for each chart
            var chartDiv = document.createElement('div');
            chartDiv.id = `chart-${year}`;
            chartDiv.className = 'chart-container';
            chartsContainer.appendChild(chartDiv);

            // Create a new div for each table
            var tableDiv = document.createElement('div');
            tableDiv.id = `table-${year}`;
            tableDiv.className = 'table-container';
            tablesContainer.appendChild(tableDiv);

            // Process the data for this year
            var regions = processPredictionData(groupedData[year]);

            // Prepare data for the chart
            var categories = ['Aman', 'Siaga', 'Awas'];
            var regionsNames = Object.keys(regions);
            var seriesData = prepareSeriesData(categories, regions, regionsNames);

            // Initialize the ECharts instance for this year
            var myChart = echarts.init(document.getElementById(`chart-${year}`));

            // Configure the ECharts option
            var option = {
                title: {
                    text: `Frekuensi Prediksi Banjir Tahun ${year}`,
                    left: 'center'
                },
                tooltip: {
                    trigger: 'axis',
                    axisPointer: {
                        type: 'shadow'
                    }
                },
                legend: {
                    data: categories,
                    top: 'bottom'
                },
                xAxis: {
                    type: 'category',
                    data: regionsNames,
                    axisLabel: {
                        rotate: 45,
                        interval: 0
                    }
                },
                yAxis: {
                    type: 'value',
                    name: 'Jumlah Prediksi'
                },
                series: seriesData
            };

            // Display the chart
            myChart.setOption(option);

            // Generate and display the table for this year
            displayDataInTable(groupedData[year], tableDiv);
        });
    })
    .catch(error => console.error('Error fetching data:', error));
}

// Fungsi untuk mengelompokkan data berdasarkan tahun
function groupDataByYear(data) {
var groupedData = {};
data.forEach(item => {
    var year = item.Tahun;
    if (!groupedData[year]) {
        groupedData[year] = [];
    }
    groupedData[year].push(item);
});
return groupedData;
}

// Fungsi untuk memproses data prediksi menjadi format yang dibutuhkan
function processPredictionData(data) {
var regions = {};
data.forEach(item => {
    if (!regions[item.Wilayah]) {
        regions[item.Wilayah] = {
            'Aman': 0,
            'Siaga': 0,
            'Awas': 0
        };
    }
    regions[item.Wilayah][item.Prediksi]++;
});
return regions;
}

// Fungsi untuk menyiapkan data seri untuk grafik
function prepareSeriesData(categories, regions, regionsNames) {
return categories.map(category => {
    return {
        name: category,
        type: 'bar',
        stack: 'total',
        data: regionsNames.map(region => regions[region][category])
    };
});
}

// Fungsi untuk menampilkan data dalam tabel
function displayDataInTable(data, container) {
var table = document.createElement('table');
table.className = 'table table-bordered';

var thead = document.createElement('thead');
thead.innerHTML = `
    <tr>
        <th>Wilayah</th>
        <th>Bulan</th>
        <th>Tahun</th>
        <th>Prediksi</th>
    </tr>
`;

var tbody = document.createElement('tbody');
data.forEach(item => {
    var row = document.createElement('tr');
    row.innerHTML = `
        <td>${item.Wilayah}</td>
        <td>${item.Bulan}</td>
        <td>${item.Tahun}</td>
        <td>${item.Prediksi}</td>
    `;
    tbody.appendChild(row);
});

table.appendChild(thead);
table.appendChild(tbody);
container.appendChild(table);
}

// Panggil fungsi utama untuk menampilkan grafik dan tabel
displayFloodPredictionChartsAndTable();
