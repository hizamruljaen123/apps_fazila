-- --------------------------------------------------------
-- Host:                         127.0.0.1
-- Server version:               8.0.30 - MySQL Community Server - GPL
-- Server OS:                    Win64
-- HeidiSQL Version:             12.1.0.6537
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

-- Dumping structure for table data_banjir.data_banjir
CREATE TABLE IF NOT EXISTS `data_banjir` (
  `Wilayah` varchar(255) DEFAULT NULL,
  `Bulan` varchar(50) DEFAULT NULL,
  `Tahun` int DEFAULT NULL,
  `Curah_Hujan` float DEFAULT NULL,
  `Suhu` float DEFAULT NULL,
  `Tinggi_Muka_Air` float DEFAULT NULL,
  `Potensi_Banjir` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ;

-- Dumping data for table data_banjir.data_banjir: ~420 rows (approximately)
INSERT INTO `data_banjir` (`Wilayah`, `Bulan`, `Tahun`, `Curah_Hujan`, `Suhu`, `Tinggi_Muka_Air`, `Potensi_Banjir`) VALUES
	('Baktiya', 'Jan', 2018, 189.4, 26.1, 277.4, 'Siaga'),
	('Lhoksukon', 'Jan', 2018, 82, 26.1, 277.4, 'Siaga'),
	('Langkahan', 'Jan', 2018, 224.5, 26.1, 277.4, 'Siaga'),
	('Cot Girek', 'Jan', 2018, 92, 26.1, 277.4, 'Siaga'),
	('Matangkuli', 'Jan', 2018, 27, 26.1, 277.4, 'Siaga'),
	('Tanah Luas', 'Jan', 2018, 54, 26.1, 277.4, 'Siaga'),
	('Stamet Aceh Utara', 'Jan', 2018, 97, 26.1, 277.4, 'Siaga'),
	('Baktiya', 'Feb', 2018, 76.7, 26.2, 100.2, 'Aman'),
	('Lhoksukon', 'Feb', 2018, 224, 26.2, 100.2, 'Waspada'),
	('Langkahan', 'Feb', 2018, 18.5, 26.2, 100.2, 'Aman'),
	('Cot Girek', 'Feb', 2018, 21, 26.2, 100.2, 'Aman'),
	('Matangkuli', 'Feb', 2018, 14, 26.2, 100.2, 'Aman'),
	('Tanah Luas', 'Feb', 2018, 20, 26.2, 100.2, 'Aman'),
	('Stamet Aceh Utara', 'Feb', 2018, 63, 26.2, 100.2, 'Aman'),
	('Baktiya', 'Mar', 2018, 13.4, 26.9, 91.9, 'Aman'),
	('Lhoksukon', 'Mar', 2018, 0, 26.9, 91.9, 'Aman'),
	('Langkahan', 'Mar', 2018, 20.5, 26.9, 91.9, 'Aman'),
	('Cot Girek', 'Mar', 2018, 4, 26.9, 91.9, 'Aman'),
	('Matangkuli', 'Mar', 2018, 2, 26.9, 91.9, 'Aman'),
	('Tanah Luas', 'Mar', 2018, 0, 26.9, 91.9, 'Aman'),
	('Stamet Aceh Utara', 'Mar', 2018, 14, 26.9, 91.9, 'Aman'),
	('Baktiya', 'Apr', 2018, 102.2, 27.2, 207.9, 'Waspada'),
	('Lhoksukon', 'Apr', 2018, 286, 27.2, 207.9, 'Waspada'),
	('Langkahan', 'Apr', 2018, 114, 27.2, 207.9, 'Waspada'),
	('Cot Girek', 'Apr', 2018, 82, 27.2, 207.9, 'Waspada'),
	('Matangkuli', 'Apr', 2018, 0, 27.2, 207.9, 'Waspada'),
	('Tanah Luas', 'Apr', 2018, 146, 27.2, 207.9, 'Waspada'),
	('Stamet Aceh Utara', 'Apr', 2018, 101, 27.2, 207.9, 'Waspada'),
	('Baktiya', 'May', 2018, 63.9, 27.2, 274.6, 'Siaga'),
	('Lhoksukon', 'May', 2018, 78, 27.2, 274.6, 'Siaga'),
	('Langkahan', 'May', 2018, 31, 27.2, 274.6, 'Siaga'),
	('Cot Girek', 'May', 2018, 181, 27.2, 274.6, 'Siaga'),
	('Matangkuli', 'May', 2018, 24, 27.2, 274.6, 'Siaga'),
	('Tanah Luas', 'May', 2018, 94, 27.2, 274.6, 'Siaga'),
	('Stamet Aceh Utara', 'May', 2018, 170, 27.2, 274.6, 'Siaga'),
	('Baktiya', 'Jun', 2018, 32.7, 27.6, 131.5, 'Aman'),
	('Lhoksukon', 'Jun', 2018, 86, 27.6, 131.5, 'Aman'),
	('Langkahan', 'Jun', 2018, 44.5, 27.6, 131.5, 'Aman'),
	('Cot Girek', 'Jun', 2018, 74, 27.6, 131.5, 'Aman'),
	('Matangkuli', 'Jun', 2018, 57, 27.6, 131.5, 'Aman'),
	('Tanah Luas', 'Jun', 2018, 95, 27.6, 131.5, 'Aman'),
	('Stamet Aceh Utara', 'Jun', 2018, 63, 27.6, 131.5, 'Aman'),
	('Baktiya', 'Jul', 2018, 168.7, 27.6, 133.1, 'Waspada'),
	('Lhoksukon', 'Jul', 2018, 384, 27.6, 133.1, 'Siaga'),
	('Langkahan', 'Jul', 2018, 86.5, 27.6, 133.1, 'Aman'),
	('Cot Girek', 'Jul', 2018, 149, 27.6, 133.1, 'Aman'),
	('Matangkuli', 'Jul', 2018, 129, 27.6, 133.1, 'Aman'),
	('Tanah Luas', 'Jul', 2018, 164, 27.6, 133.1, 'Waspada'),
	('Stamet Aceh Utara', 'Jul', 2018, 120, 27.6, 133.1, 'Aman'),
	('Baktiya', 'Aug', 2018, 22, 27.2, 47.9, 'Aman'),
	('Lhoksukon', 'Aug', 2018, 212, 27.2, 47.9, 'Waspada'),
	('Langkahan', 'Aug', 2018, 54.5, 27.2, 47.9, 'Aman'),
	('Cot Girek', 'Aug', 2018, 181, 27.2, 47.9, 'Waspada'),
	('Matangkuli', 'Aug', 2018, 90, 27.2, 47.9, 'Aman'),
	('Tanah Luas', 'Aug', 2018, 114, 27.2, 47.9, 'Aman'),
	('Stamet Aceh Utara', 'Aug', 2018, 79, 27.2, 47.9, 'Aman'),
	('Baktiya', 'Sep', 2018, 199.5, 26.5, 266.7, 'Siaga'),
	('Lhoksukon', 'Sep', 2018, 432, 26.5, 266.7, 'Awas'),
	('Langkahan', 'Sep', 2018, 138, 26.5, 266.7, 'Siaga'),
	('Cot Girek', 'Sep', 2018, 128, 26.5, 266.7, 'Siaga'),
	('Matangkuli', 'Sep', 2018, 227, 26.5, 266.7, 'Siaga'),
	('Tanah Luas', 'Sep', 2018, 216, 26.5, 266.7, 'Siaga'),
	('Stamet Aceh Utara', 'Sep', 2018, 97, 26.5, 266.7, 'Siaga'),
	('Baktiya', 'Oct', 2018, 269.7, 26.3, 370.6, 'Awas'),
	('Lhoksukon', 'Oct', 2018, 488, 26.3, 370.6, 'Awas'),
	('Langkahan', 'Oct', 2018, 236, 26.3, 370.6, 'Awas'),
	('Cot Girek', 'Oct', 2018, 168, 26.3, 370.6, 'Awas'),
	('Matangkuli', 'Oct', 2018, 363, 26.3, 370.6, 'Awas'),
	('Tanah Luas', 'Oct', 2018, 320, 26.3, 370.6, 'Awas'),
	('Stamet Aceh Utara', 'Oct', 2018, 258, 26.3, 370.6, 'Awas'),
	('Baktiya', 'Nov', 2018, 214.2, 26.2, 369.4, 'Awas'),
	('Lhoksukon', 'Nov', 2018, 534, 26.2, 369.4, 'Awas'),
	('Langkahan', 'Nov', 2018, 223, 26.2, 369.4, 'Awas'),
	('Cot Girek', 'Nov', 2018, 64, 26.2, 369.4, 'Awas'),
	('Matangkuli', 'Nov', 2018, 268, 26.2, 369.4, 'Awas'),
	('Tanah Luas', 'Nov', 2018, 243, 26.2, 369.4, 'Awas'),
	('Stamet Aceh Utara', 'Nov', 2018, 296, 26.2, 369.4, 'Awas'),
	('Baktiya', 'Dec', 2018, 45.3, 26.1, 260.8, 'Siaga'),
	('Lhoksukon', 'Dec', 2018, 68, 26.1, 260.8, 'Siaga'),
	('Langkahan', 'Dec', 2018, 88.5, 26.1, 260.8, 'Siaga'),
	('Cot Girek', 'Dec', 2018, 64, 26.1, 260.8, 'Siaga'),
	('Matangkuli', 'Dec', 2018, 56, 26.1, 260.8, 'Siaga'),
	('Tanah Luas', 'Dec', 2018, 40, 26.1, 260.8, 'Siaga'),
	('Stamet Aceh Utara', 'Dec', 2018, 74, 26.1, 260.8, 'Siaga'),
	('Baktiya', 'Jan', 2019, 76, 26.4, 267.9, 'Siaga'),
	('Lhoksukon', 'Jan', 2019, 102, 26.4, 267.9, 'Siaga'),
	('Langkahan', 'Jan', 2019, 95.5, 26.4, 267.9, 'Siaga'),
	('Cot Girek', 'Jan', 2019, 91, 26.4, 267.9, 'Siaga'),
	('Matangkuli', 'Jan', 2019, 74, 26.4, 267.9, 'Siaga'),
	('Tanah Luas', 'Jan', 2019, 109, 26.4, 267.9, 'Siaga'),
	('Stamet Aceh Utara', 'Jan', 2019, 40, 26.4, 267.9, 'Siaga'),
	('Baktiya', 'Feb', 2019, 41, 26.6, 274.2, 'Siaga'),
	('Lhoksukon', 'Feb', 2019, 5, 26.6, 274.2, 'Siaga'),
	('Langkahan', 'Feb', 2019, 38, 26.6, 274.2, 'Siaga'),
	('Cot Girek', 'Feb', 2019, 20, 26.6, 274.2, 'Siaga'),
	('Matangkuli', 'Feb', 2019, 31, 26.6, 274.2, 'Siaga'),
	('Tanah Luas', 'Feb', 2019, 50, 26.6, 274.2, 'Siaga'),
	('Stamet Aceh Utara', 'Feb', 2019, 43, 26.6, 274.2, 'Siaga'),
	('Baktiya', 'Mar', 2019, 72, 27.3, 214.9, 'Waspada'),
	('Lhoksukon', 'Mar', 2019, 15, 27.3, 214.9, 'Waspada'),
	('Langkahan', 'Mar', 2019, 20, 27.3, 214.9, 'Waspada'),
	('Cot Girek', 'Mar', 2019, 15, 27.3, 214.9, 'Waspada'),
	('Matangkuli', 'Mar', 2019, 36, 27.3, 214.9, 'Waspada'),
	('Tanah Luas', 'Mar', 2019, 26, 27.3, 214.9, 'Waspada'),
	('Stamet Aceh Utara', 'Mar', 2019, 40, 27.3, 214.9, 'Waspada'),
	('Baktiya', 'Apr', 2019, 64, 27.4, 239.1, 'Waspada'),
	('Lhoksukon', 'Apr', 2019, 103, 27.4, 239.1, 'Waspada'),
	('Langkahan', 'Apr', 2019, 97, 27.4, 239.1, 'Waspada'),
	('Cot Girek', 'Apr', 2019, 85, 27.4, 239.1, 'Waspada'),
	('Matangkuli', 'Apr', 2019, 84, 27.4, 239.1, 'Waspada'),
	('Tanah Luas', 'Apr', 2019, 65, 27.4, 239.1, 'Waspada'),
	('Stamet Aceh Utara', 'Apr', 2019, 45, 27.4, 239.1, 'Waspada'),
	('Baktiya', 'May', 2019, 99, 27.7, 294.7, 'Siaga'),
	('Lhoksukon', 'May', 2019, 158, 27.7, 294.7, 'Siaga'),
	('Langkahan', 'May', 2019, 116.5, 27.7, 294.7, 'Siaga'),
	('Cot Girek', 'May', 2019, 24, 27.7, 294.7, 'Siaga'),
	('Matangkuli', 'May', 2019, 83, 27.7, 294.7, 'Siaga'),
	('Tanah Luas', 'May', 2019, 141, 27.7, 294.7, 'Siaga'),
	('Stamet Aceh Utara', 'May', 2019, 121, 27.7, 294.7, 'Siaga'),
	('Baktiya', 'Jun', 2019, 96, 27.7, 154.1, 'Waspada'),
	('Lhoksukon', 'Jun', 2019, 189, 27.7, 154.1, 'Waspada'),
	('Langkahan', 'Jun', 2019, 62, 27.7, 154.1, 'Waspada'),
	('Cot Girek', 'Jun', 2019, 111, 27.7, 154.1, 'Waspada'),
	('Matangkuli', 'Jun', 2019, 134, 27.7, 154.1, 'Waspada'),
	('Tanah Luas', 'Jun', 2019, 92, 27.7, 154.1, 'Waspada'),
	('Stamet Aceh Utara', 'Jun', 2019, 81, 27.7, 154.1, 'Waspada'),
	('Baktiya', 'Jul', 2019, 4, 27.4, 183.2, 'Waspada'),
	('Lhoksukon', 'Jul', 2019, 50, 27.4, 183.2, 'Waspada'),
	('Langkahan', 'Jul', 2019, 61.5, 27.4, 183.2, 'Waspada'),
	('Cot Girek', 'Jul', 2019, 47, 27.4, 183.2, 'Waspada'),
	('Matangkuli', 'Jul', 2019, 116, 27.4, 183.2, 'Waspada'),
	('Tanah Luas', 'Jul', 2019, 74, 27.4, 183.2, 'Waspada'),
	('Stamet Aceh Utara', 'Jul', 2019, 54, 27.4, 183.2, 'Waspada'),
	('Baktiya', 'Aug', 2019, 44, 27.3, 204.6, 'Waspada'),
	('Lhoksukon', 'Aug', 2019, 122, 27.3, 204.6, 'Waspada'),
	('Langkahan', 'Aug', 2019, 126, 27.3, 204.6, 'Waspada'),
	('Cot Girek', 'Aug', 2019, 54, 27.3, 204.6, 'Waspada'),
	('Matangkuli', 'Aug', 2019, 164, 27.3, 204.6, 'Waspada'),
	('Tanah Luas', 'Aug', 2019, 284, 27.3, 204.6, 'Waspada'),
	('Stamet Aceh Utara', 'Aug', 2019, 81, 27.3, 204.6, 'Waspada'),
	('Baktiya', 'Sep', 2019, 124, 26.7, 375.6, 'Awas'),
	('Lhoksukon', 'Sep', 2019, 128, 26.7, 375.6, 'Awas'),
	('Langkahan', 'Sep', 2019, 176.5, 26.7, 375.6, 'Awas'),
	('Cot Girek', 'Sep', 2019, 169, 26.7, 375.6, 'Awas'),
	('Matangkuli', 'Sep', 2019, 152, 26.7, 375.6, 'Awas'),
	('Tanah Luas', 'Sep', 2019, 184, 26.7, 375.6, 'Awas'),
	('Stamet Aceh Utara', 'Sep', 2019, 106, 26.7, 375.6, 'Awas'),
	('Baktiya', 'Oct', 2019, 170, 25.9, 272.4, 'Siaga'),
	('Lhoksukon', 'Oct', 2019, 143, 25.9, 272.4, 'Siaga'),
	('Langkahan', 'Oct', 2019, 338, 25.9, 272.4, 'Siaga'),
	('Cot Girek', 'Oct', 2019, 131, 25.9, 272.4, 'Siaga'),
	('Matangkuli', 'Oct', 2019, 168, 25.9, 272.4, 'Siaga'),
	('Tanah Luas', 'Oct', 2019, 152, 25.9, 272.4, 'Siaga'),
	('Stamet Aceh Utara', 'Oct', 2019, 123, 25.9, 272.4, 'Siaga'),
	('Baktiya', 'Nov', 2019, 180, 26.2, 261.6, 'Siaga'),
	('Lhoksukon', 'Nov', 2019, 96, 26.2, 261.6, 'Siaga'),
	('Langkahan', 'Nov', 2019, 124.5, 26.2, 261.6, 'Siaga'),
	('Cot Girek', 'Nov', 2019, 138, 26.2, 261.6, 'Siaga'),
	('Matangkuli', 'Nov', 2019, 93, 26.2, 261.6, 'Siaga'),
	('Tanah Luas', 'Nov', 2019, 51, 26.2, 261.6, 'Siaga'),
	('Stamet Aceh Utara', 'Nov', 2019, 109, 26.2, 261.6, 'Siaga'),
	('Baktiya', 'Dec', 2019, 441, 25.8, 374.8, 'Awas'),
	('Lhoksukon', 'Dec', 2019, 194, 25.8, 374.8, 'Awas'),
	('Langkahan', 'Dec', 2019, 144.5, 25.8, 374.8, 'Awas'),
	('Cot Girek', 'Dec', 2019, 134, 25.8, 374.8, 'Awas'),
	('Matangkuli', 'Dec', 2019, 186, 25.8, 374.8, 'Awas'),
	('Tanah Luas', 'Dec', 2019, 199, 25.8, 374.8, 'Awas'),
	('Stamet Aceh Utara', 'Dec', 2019, 128, 25.8, 374.8, 'Awas'),
	('Baktiya', 'Jan', 2020, 44, 26.6, 100.2, 'Aman'),
	('Lhoksukon', 'Jan', 2020, 10, 26.6, 100.2, 'Aman'),
	('Langkahan', 'Jan', 2020, 47.5, 26.6, 100.2, 'Aman'),
	('Cot Girek', 'Jan', 2020, 15, 26.6, 100.2, 'Aman'),
	('Matangkuli', 'Jan', 2020, 26, 26.6, 100.2, 'Aman'),
	('Tanah Luas', 'Jan', 2020, 28, 26.6, 100.2, 'Aman'),
	('Stamet Aceh Utara', 'Jan', 2020, 7, 26.6, 100.2, 'Aman'),
	('Baktiya', 'Feb', 2020, 155.4, 26.5, 162.4, 'Waspada'),
	('Lhoksukon', 'Feb', 2020, 36, 26.5, 162.4, 'Waspada'),
	('Langkahan', 'Feb', 2020, 75.5, 26.5, 162.4, 'Waspada'),
	('Cot Girek', 'Feb', 2020, 25, 26.5, 162.4, 'Waspada'),
	('Matangkuli', 'Feb', 2020, 45, 26.5, 162.4, 'Waspada'),
	('Tanah Luas', 'Feb', 2020, 36, 26.5, 162.4, 'Waspada'),
	('Stamet Aceh Utara', 'Feb', 2020, 56, 26.5, 162.4, 'Waspada'),
	('Baktiya', 'Mar', 2020, 47.4, 27.4, 182.4, 'Waspada'),
	('Lhoksukon', 'Mar', 2020, 16, 27.4, 182.4, 'Waspada'),
	('Langkahan', 'Mar', 2020, 29, 27.4, 182.4, 'Waspada'),
	('Cot Girek', 'Mar', 2020, 10, 27.4, 182.4, 'Waspada'),
	('Matangkuli', 'Mar', 2020, 22, 27.4, 182.4, 'Waspada'),
	('Tanah Luas', 'Mar', 2020, 29, 27.4, 182.4, 'Waspada'),
	('Stamet Aceh Utara', 'Mar', 2020, 20, 27.4, 182.4, 'Waspada'),
	('Baktiya', 'Apr', 2020, 50.8, 27.3, 299.4, 'Siaga'),
	('Lhoksukon', 'Apr', 2020, 155, 27.3, 299.4, 'Siaga'),
	('Langkahan', 'Apr', 2020, 8, 27.3, 299.4, 'Siaga'),
	('Cot Girek', 'Apr', 2020, 43, 27.3, 299.4, 'Siaga'),
	('Matangkuli', 'Apr', 2020, 68, 27.3, 299.4, 'Siaga'),
	('Tanah Luas', 'Apr', 2020, 152, 27.3, 299.4, 'Siaga'),
	('Stamet Aceh Utara', 'Apr', 2020, 249, 27.3, 299.4, 'Siaga'),
	('Baktiya', 'May', 2020, 240.3, 27.2, 286.2, 'Siaga'),
	('Lhoksukon', 'May', 2020, 430, 27.2, 286.2, 'Awas'),
	('Langkahan', 'May', 2020, 342, 27.2, 286.2, 'Siaga'),
	('Cot Girek', 'May', 2020, 471, 27.2, 286.2, 'Awas'),
	('Matangkuli', 'May', 2020, 475, 27.2, 286.2, 'Awas'),
	('Tanah Luas', 'May', 2020, 489, 27.2, 286.2, 'Awas'),
	('Stamet Aceh Utara', 'May', 2020, 434, 27.2, 286.2, 'Awas'),
	('Baktiya', 'Jun', 2020, 139.2, 27.1, 211.6, 'Waspada'),
	('Lhoksukon', 'Jun', 2020, 231, 27.1, 211.6, 'Waspada'),
	('Langkahan', 'Jun', 2020, 17, 27.1, 211.6, 'Waspada'),
	('Cot Girek', 'Jun', 2020, 200, 27.1, 211.6, 'Waspada'),
	('Matangkuli', 'Jun', 2020, 191, 27.1, 211.6, 'Waspada'),
	('Tanah Luas', 'Jun', 2020, 195, 27.1, 211.6, 'Waspada'),
	('Stamet Aceh Utara', 'Jun', 2020, 141, 27.1, 211.6, 'Waspada'),
	('Baktiya', 'Jul', 2020, 59, 26.7, 253.2, 'Siaga'),
	('Lhoksukon', 'Jul', 2020, 287, 26.7, 253.2, 'Siaga'),
	('Langkahan', 'Jul', 2020, 91.5, 26.7, 253.2, 'Siaga'),
	('Cot Girek', 'Jul', 2020, 302, 26.7, 253.2, 'Siaga'),
	('Matangkuli', 'Jul', 2020, 273, 26.7, 253.2, 'Siaga'),
	('Tanah Luas', 'Jul', 2020, 233, 26.7, 253.2, 'Siaga'),
	('Stamet Aceh Utara', 'Jul', 2020, 178, 26.7, 253.2, 'Siaga'),
	('Baktiya', 'Aug', 2020, 176.3, 27.4, 244.6, 'Waspada'),
	('Lhoksukon', 'Aug', 2020, 98, 27.4, 244.6, 'Waspada'),
	('Langkahan', 'Aug', 2020, 70, 27.4, 244.6, 'Waspada'),
	('Cot Girek', 'Aug', 2020, 180, 27.4, 244.6, 'Waspada'),
	('Matangkuli', 'Aug', 2020, 137, 27.4, 244.6, 'Waspada'),
	('Tanah Luas', 'Aug', 2020, 108, 27.4, 244.6, 'Waspada'),
	('Stamet Aceh Utara', 'Aug', 2020, 3, 27.4, 244.6, 'Waspada'),
	('Baktiya', 'Sep', 2020, 270.6, 26.9, 235.1, 'Waspada'),
	('Lhoksukon', 'Sep', 2020, 186, 26.9, 235.1, 'Waspada'),
	('Langkahan', 'Sep', 2020, 34, 26.9, 235.1, 'Waspada'),
	('Cot Girek', 'Sep', 2020, 88, 26.9, 235.1, 'Waspada'),
	('Matangkuli', 'Sep', 2020, 157, 26.9, 235.1, 'Waspada'),
	('Tanah Luas', 'Sep', 2020, 169, 26.9, 235.1, 'Waspada'),
	('Stamet Aceh Utara', 'Sep', 2020, 64, 26.9, 235.1, 'Waspada'),
	('Baktiya', 'Oct', 2020, 251.1, 26.9, 247.9, 'Waspada'),
	('Lhoksukon', 'Oct', 2020, 90, 26.9, 247.9, 'Waspada'),
	('Langkahan', 'Oct', 2020, 117.5, 26.9, 247.9, 'Waspada'),
	('Cot Girek', 'Oct', 2020, 83, 26.9, 247.9, 'Waspada'),
	('Matangkuli', 'Oct', 2020, 174, 26.9, 247.9, 'Waspada'),
	('Tanah Luas', 'Oct', 2020, 120, 26.9, 247.9, 'Waspada'),
	('Stamet Aceh Utara', 'Oct', 2020, 75, 26.9, 247.9, 'Waspada'),
	('Baktiya', 'Nov', 2020, 180.7, 26.2, 361.3, 'Awas'),
	('Lhoksukon', 'Nov', 2020, 116, 26.2, 361.3, 'Awas'),
	('Langkahan', 'Nov', 2020, 167.5, 26.2, 361.3, 'Awas'),
	('Cot Girek', 'Nov', 2020, 177, 26.2, 361.3, 'Awas'),
	('Matangkuli', 'Nov', 2020, 150, 26.2, 361.3, 'Awas'),
	('Tanah Luas', 'Nov', 2020, 176, 26.2, 361.3, 'Awas'),
	('Stamet Aceh Utara', 'Nov', 2020, 226, 26.2, 361.3, 'Awas'),
	('Baktiya', 'Dec', 2020, 544.7, 25.7, 633.2, 'Awas'),
	('Lhoksukon', 'Dec', 2020, 438, 25.7, 633.2, 'Awas'),
	('Langkahan', 'Dec', 2020, 376.5, 25.7, 633.2, 'Awas'),
	('Cot Girek', 'Dec', 2020, 690, 25.7, 633.2, 'Awas'),
	('Matangkuli', 'Dec', 2020, 422, 25.7, 633.2, 'Awas'),
	('Tanah Luas', 'Dec', 2020, 538, 25.7, 633.2, 'Awas'),
	('Stamet Aceh Utara', 'Dec', 2020, 375, 25.7, 633.2, 'Awas'),
	('Baktiya', 'Jan', 2021, 515, 25.8, 0, 'Awas'),
	('Lhoksukon', 'Jan', 2021, 219, 25.8, 0, 'Waspada'),
	('Langkahan', 'Jan', 2021, 273, 25.8, 0, 'Waspada'),
	('Cot Girek', 'Jan', 2021, 334, 25.8, 0, 'Siaga'),
	('Matangkuli', 'Jan', 2021, 203, 25.8, 0, 'Waspada'),
	('Tanah Luas', 'Jan', 2021, 177, 25.8, 0, 'Waspada'),
	('Stamet Aceh Utara', 'Jan', 2021, 275, 25.8, 0, 'Waspada'),
	('Baktiya', 'Feb', 2021, 2, 26.4, 0, 'Aman'),
	('Lhoksukon', 'Feb', 2021, 1, 26.4, 0, 'Aman'),
	('Langkahan', 'Feb', 2021, 12, 26.4, 0, 'Aman'),
	('Cot Girek', 'Feb', 2021, 7, 26.4, 0, 'Aman'),
	('Matangkuli', 'Feb', 2021, 23, 26.4, 0, 'Aman'),
	('Tanah Luas', 'Feb', 2021, 0, 26.4, 0, 'Aman'),
	('Stamet Aceh Utara', 'Feb', 2021, 49, 26.4, 0, 'Aman'),
	('Baktiya', 'Mar', 2021, 111, 26.6, 0, 'Aman'),
	('Lhoksukon', 'Mar', 2021, 53, 26.6, 0, 'Aman'),
	('Langkahan', 'Mar', 2021, 176, 26.6, 0, 'Waspada'),
	('Cot Girek', 'Mar', 2021, 40, 26.6, 0, 'Aman'),
	('Matangkuli', 'Mar', 2021, 160, 26.6, 0, 'Waspada'),
	('Tanah Luas', 'Mar', 2021, 0, 26.6, 0, 'Aman'),
	('Stamet Aceh Utara', 'Mar', 2021, 122, 26.6, 0, 'Aman'),
	('Baktiya', 'Apr', 2021, 241, 26.8, 0, 'Waspada'),
	('Lhoksukon', 'Apr', 2021, 170, 26.8, 0, 'Waspada'),
	('Langkahan', 'Apr', 2021, 116, 26.8, 0, 'Aman'),
	('Cot Girek', 'Apr', 2021, 208, 26.8, 0, 'Waspada'),
	('Matangkuli', 'Apr', 2021, 138, 26.8, 0, 'Aman'),
	('Tanah Luas', 'Apr', 2021, 114, 26.8, 0, 'Aman'),
	('Stamet Aceh Utara', 'Apr', 2021, 117, 26.8, 0, 'Aman'),
	('Baktiya', 'May', 2021, 41, 27.4, 0, 'Aman'),
	('Lhoksukon', 'May', 2021, 135, 27.4, 0, 'Aman'),
	('Langkahan', 'May', 2021, 134, 27.4, 0, 'Aman'),
	('Cot Girek', 'May', 2021, 131, 27.4, 0, 'Aman'),
	('Matangkuli', 'May', 2021, 131, 27.4, 0, 'Aman'),
	('Tanah Luas', 'May', 2021, 91, 27.4, 0, 'Aman'),
	('Stamet Aceh Utara', 'May', 2021, 137, 27.4, 0, 'Aman'),
	('Baktiya', 'Jun', 2021, 69, 26.9, 0, 'Aman'),
	('Lhoksukon', 'Jun', 2021, 99, 26.9, 0, 'Aman'),
	('Langkahan', 'Jun', 2021, 34, 26.9, 0, 'Aman'),
	('Cot Girek', 'Jun', 2021, 123, 26.9, 0, 'Aman'),
	('Matangkuli', 'Jun', 2021, 78, 26.9, 0, 'Aman'),
	('Tanah Luas', 'Jun', 2021, 125, 26.9, 0, 'Aman'),
	('Stamet Aceh Utara', 'Jun', 2021, 215, 26.9, 0, 'Waspada'),
	('Baktiya', 'Jul', 2021, 80, 27.2, 0, 'Aman'),
	('Lhoksukon', 'Jul', 2021, 124, 27.2, 0, 'Aman'),
	('Langkahan', 'Jul', 2021, 70, 27.2, 0, 'Aman'),
	('Cot Girek', 'Jul', 2021, 253, 27.2, 0, 'Waspada'),
	('Matangkuli', 'Jul', 2021, 101, 27.2, 0, 'Aman'),
	('Tanah Luas', 'Jul', 2021, 74, 27.2, 0, 'Aman'),
	('Stamet Aceh Utara', 'Jul', 2021, 72, 27.2, 0, 'Aman'),
	('Baktiya', 'Aug', 2021, 175, 26.7, 0, 'Waspada'),
	('Lhoksukon', 'Aug', 2021, 372, 26.7, 0, 'Siaga'),
	('Langkahan', 'Aug', 2021, 330, 26.7, 0, 'Siaga'),
	('Cot Girek', 'Aug', 2021, 352, 26.7, 0, 'Siaga'),
	('Matangkuli', 'Aug', 2021, 268, 26.7, 0, 'Waspada'),
	('Tanah Luas', 'Aug', 2021, 51, 26.7, 0, 'Aman'),
	('Stamet Aceh Utara', 'Aug', 2021, 153, 26.7, 0, 'Waspada'),
	('Baktiya', 'Sep', 2021, 144, 26.7, 0, 'Aman'),
	('Lhoksukon', 'Sep', 2021, 78, 26.7, 0, 'Aman'),
	('Langkahan', 'Sep', 2021, 86, 26.7, 0, 'Aman'),
	('Cot Girek', 'Sep', 2021, 261, 26.7, 0, 'Waspada'),
	('Matangkuli', 'Sep', 2021, 61, 26.7, 0, 'Aman'),
	('Tanah Luas', 'Sep', 2021, 92, 26.7, 0, 'Aman'),
	('Stamet Aceh Utara', 'Sep', 2021, 42, 26.7, 0, 'Aman'),
	('Baktiya', 'Oct', 2021, 186, 26.6, 0, 'Waspada'),
	('Lhoksukon', 'Oct', 2021, 120, 26.6, 0, 'Aman'),
	('Langkahan', 'Oct', 2021, 158, 26.6, 0, 'Waspada'),
	('Cot Girek', 'Oct', 2021, 138, 26.6, 0, 'Aman'),
	('Matangkuli', 'Oct', 2021, 150, 26.6, 0, 'Aman'),
	('Tanah Luas', 'Oct', 2021, 125, 26.6, 0, 'Aman'),
	('Stamet Aceh Utara', 'Oct', 2021, 199, 26.6, 0, 'Waspada'),
	('Baktiya', 'Nov', 2021, 326, 25.9, 0, 'Siaga'),
	('Lhoksukon', 'Nov', 2021, 332, 25.9, 0, 'Siaga'),
	('Langkahan', 'Nov', 2021, 150, 25.9, 0, 'Aman'),
	('Cot Girek', 'Nov', 2021, 340, 25.9, 0, 'Siaga'),
	('Matangkuli', 'Nov', 2021, 313, 25.9, 0, 'Siaga'),
	('Tanah Luas', 'Nov', 2021, 200, 25.9, 0, 'Waspada'),
	('Stamet Aceh Utara', 'Nov', 2021, 161, 25.9, 0, 'Waspada'),
	('Baktiya', 'Dec', 2021, 787, 25.8, 0, 'Awas'),
	('Lhoksukon', 'Dec', 2021, 215, 25.8, 0, 'Waspada'),
	('Langkahan', 'Dec', 2021, 273, 25.8, 0, 'Waspada'),
	('Cot Girek', 'Dec', 2021, 515, 25.8, 0, 'Awas'),
	('Matangkuli', 'Dec', 2021, 193, 25.8, 0, 'Waspada'),
	('Tanah Luas', 'Dec', 2021, 291, 25.8, 0, 'Waspada'),
	('Stamet Aceh Utara', 'Dec', 2021, 232, 25.8, 0, 'Waspada'),
	('Baktiya', 'Jan', 2022, 218, 26.2, 504, 'Awas'),
	('Lhoksukon', 'Jan', 2022, 224, 26.2, 504, 'Awas'),
	('Langkahan', 'Jan', 2022, 212, 26.2, 504, 'Awas'),
	('Cot Girek', 'Jan', 2022, 172, 26.2, 504, 'Awas'),
	('Matangkuli', 'Jan', 2022, 182, 26.2, 504, 'Awas'),
	('Tanah Luas', 'Jan', 2022, 90, 26.2, 504, 'Awas'),
	('Stamet Aceh Utara', 'Jan', 2022, 129, 26.2, 504, 'Awas'),
	('Baktiya', 'Feb', 2022, 533, 25.9, 399.9, 'Awas'),
	('Lhoksukon', 'Feb', 2022, 322, 25.9, 399.9, 'Awas'),
	('Langkahan', 'Feb', 2022, 467, 25.9, 399.9, 'Awas'),
	('Cot Girek', 'Feb', 2022, 263, 25.9, 399.9, 'Awas'),
	('Matangkuli', 'Feb', 2022, 304, 25.9, 399.9, 'Awas'),
	('Tanah Luas', 'Feb', 2022, 176, 25.9, 399.9, 'Awas'),
	('Stamet Aceh Utara', 'Feb', 2022, 295, 25.9, 399.9, 'Awas'),
	('Baktiya', 'Mar', 2022, 109, 26.4, 493.9, 'Awas'),
	('Lhoksukon', 'Mar', 2022, 91, 26.4, 493.9, 'Awas'),
	('Langkahan', 'Mar', 2022, 54, 26.4, 493.9, 'Awas'),
	('Cot Girek', 'Mar', 2022, 146, 26.4, 493.9, 'Awas'),
	('Matangkuli', 'Mar', 2022, 95, 26.4, 493.9, 'Awas'),
	('Tanah Luas', 'Mar', 2022, 55, 26.4, 493.9, 'Awas'),
	('Stamet Aceh Utara', 'Mar', 2022, 60, 26.4, 493.9, 'Awas'),
	('Baktiya', 'Apr', 2022, 68, 26.6, 498.4, 'Awas'),
	('Lhoksukon', 'Apr', 2022, 11, 26.6, 498.4, 'Awas'),
	('Langkahan', 'Apr', 2022, 30, 26.6, 498.4, 'Awas'),
	('Cot Girek', 'Apr', 2022, 5, 26.6, 498.4, 'Awas'),
	('Matangkuli', 'Apr', 2022, 48, 26.6, 498.4, 'Awas'),
	('Tanah Luas', 'Apr', 2022, 25, 26.6, 498.4, 'Awas'),
	('Stamet Aceh Utara', 'Apr', 2022, 110, 26.6, 498.4, 'Awas'),
	('Baktiya', 'May', 2022, 164, 27.1, 508.6, 'Awas'),
	('Lhoksukon', 'May', 2022, 270, 27.1, 508.6, 'Awas'),
	('Langkahan', 'May', 2022, 125, 27.1, 508.6, 'Awas'),
	('Cot Girek', 'May', 2022, 172, 27.1, 508.6, 'Awas'),
	('Matangkuli', 'May', 2022, 214, 27.1, 508.6, 'Awas'),
	('Tanah Luas', 'May', 2022, 200, 27.1, 508.6, 'Awas'),
	('Stamet Aceh Utara', 'May', 2022, 63, 27.1, 508.6, 'Awas'),
	('Baktiya', 'Jun', 2022, 160, 26.5, 219.1, 'Waspada'),
	('Lhoksukon', 'Jun', 2022, 371, 26.5, 219.1, 'Siaga'),
	('Langkahan', 'Jun', 2022, 145, 26.5, 219.1, 'Waspada'),
	('Cot Girek', 'Jun', 2022, 270, 26.5, 219.1, 'Waspada'),
	('Matangkuli', 'Jun', 2022, 246, 26.5, 219.1, 'Waspada'),
	('Tanah Luas', 'Jun', 2022, 200, 26.5, 219.1, 'Waspada'),
	('Stamet Aceh Utara', 'Jun', 2022, 186, 26.5, 219.1, 'Waspada'),
	('Baktiya', 'Jul', 2022, 75, 27.2, 183.8, 'Waspada'),
	('Lhoksukon', 'Jul', 2022, 24, 27.2, 183.8, 'Waspada'),
	('Langkahan', 'Jul', 2022, 77, 27.2, 183.8, 'Waspada'),
	('Cot Girek', 'Jul', 2022, 177, 27.2, 183.8, 'Waspada'),
	('Matangkuli', 'Jul', 2022, 141, 27.2, 183.8, 'Waspada'),
	('Tanah Luas', 'Jul', 2022, 57, 27.2, 183.8, 'Waspada'),
	('Stamet Aceh Utara', 'Jul', 2022, 31, 27.2, 183.8, 'Waspada'),
	('Baktiya', 'Aug', 2022, 129, 26.8, 252.5, 'Siaga'),
	('Lhoksukon', 'Aug', 2022, 197, 26.8, 252.5, 'Siaga'),
	('Langkahan', 'Aug', 2022, 161, 26.8, 252.5, 'Siaga'),
	('Cot Girek', 'Aug', 2022, 292, 26.8, 252.5, 'Siaga'),
	('Matangkuli', 'Aug', 2022, 259, 26.8, 252.5, 'Siaga'),
	('Tanah Luas', 'Aug', 2022, 304, 26.8, 252.5, 'Siaga'),
	('Stamet Aceh Utara', 'Aug', 2022, 94, 26.8, 252.5, 'Siaga'),
	('Baktiya', 'Sep', 2022, 211, 26.7, 253.2, 'Siaga'),
	('Lhoksukon', 'Sep', 2022, 246, 26.7, 253.2, 'Siaga'),
	('Langkahan', 'Sep', 2022, 165, 26.7, 253.2, 'Siaga'),
	('Cot Girek', 'Sep', 2022, 396, 26.7, 253.2, 'Siaga'),
	('Matangkuli', 'Sep', 2022, 188, 26.7, 253.2, 'Siaga'),
	('Tanah Luas', 'Sep', 2022, 211, 26.7, 253.2, 'Siaga'),
	('Stamet Aceh Utara', 'Sep', 2022, 55, 26.7, 253.2, 'Siaga'),
	('Baktiya', 'Oct', 2022, 240, 26.1, 452.6, 'Awas'),
	('Lhoksukon', 'Oct', 2022, 300, 26.1, 452.6, 'Awas'),
	('Langkahan', 'Oct', 2022, 289, 26.1, 452.6, 'Awas'),
	('Cot Girek', 'Oct', 2022, 580, 26.1, 452.6, 'Awas'),
	('Matangkuli', 'Oct', 2022, 305, 26.1, 452.6, 'Awas'),
	('Tanah Luas', 'Oct', 2022, 307, 26.1, 452.6, 'Awas'),
	('Stamet Aceh Utara', 'Oct', 2022, 358, 26.1, 452.6, 'Awas'),
	('Baktiya', 'Nov', 2022, 164, 25.8, 352.8, 'Awas'),
	('Lhoksukon', 'Nov', 2022, 245, 25.8, 352.8, 'Awas'),
	('Langkahan', 'Nov', 2022, 229, 25.8, 352.8, 'Awas'),
	('Cot Girek', 'Nov', 2022, 338, 25.8, 352.8, 'Awas'),
	('Matangkuli', 'Nov', 2022, 218, 25.8, 352.8, 'Awas'),
	('Tanah Luas', 'Nov', 2022, 274, 25.8, 352.8, 'Awas'),
	('Stamet Aceh Utara', 'Nov', 2022, 270, 25.8, 352.8, 'Awas'),
	('Baktiya', 'Dec', 2022, 380, 25.4, 387.5, 'Awas'),
	('Lhoksukon', 'Dec', 2022, 417, 25.4, 387.5, 'Awas'),
	('Langkahan', 'Dec', 2022, 344, 25.4, 387.5, 'Awas'),
	('Cot Girek', 'Dec', 2022, 339, 25.4, 387.5, 'Awas'),
	('Matangkuli', 'Dec', 2022, 340, 25.4, 387.5, 'Awas'),
	('Tanah Luas', 'Dec', 2022, 346, 25.4, 387.5, 'Awas'),
	('Stamet Aceh Utara', 'Dec', 2022, 374, 25.4, 387.5, 'Awas');

-- Dumping structure for table data_banjir.data_uji
CREATE TABLE IF NOT EXISTS `data_uji` (
  `Wilayah` varchar(255)  NULL,
  `Bulan` varchar(50)  NULL,
  `Tahun` int DEFAULT NULL,
  `Curah_Hujan` float DEFAULT NULL,
  `Suhu` float DEFAULT NULL,
  `Tinggi_Muka_Air` float DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 ;

-- Dumping data for table data_banjir.data_uji: ~17 rows (approximately)
INSERT INTO `data_uji` (`Wilayah`, `Bulan`, `Tahun`, `Curah_Hujan`, `Suhu`, `Tinggi_Muka_Air`) VALUES
	('Lhoksukon', 'Januari', 2023, 310.32, 33.66, 0.55),
	('Tanah Luas', 'Februari', 2023, 296.81, 21.21, 2.64),
	('Stamet Aceh Utara', 'Maret', 2023, 116, 23.88, 1.44),
	('Langkahan', 'April', 2023, 251.85, 25.08, 1.26),
	('Tanah Luas', 'Mei', 2023, 208.53, 30.18, 2.12),
	('Matangkuli', 'Juni', 2023, 319.43, 23.35, 0.63),
	('Tanah Luas', 'Juli', 2023, 248.73, 22.98, 1.52),
	('Matangkuli', 'Agustus', 2023, 319.18, 30.61, 1.35),
	('Baktiya', 'September', 2023, 374.07, 25.07, 2.87),
	('Tanah Luas', 'Januari', 2024, 292.75, 33.61, 2.39),
	('Lhoksukon', 'Februari', 2024, 367.08, 29.48, 2.41),
	('Tanah Luas', 'Maret', 2024, 138.47, 31.13, 1.72),
	('Tanah Luas', 'April', 2024, 179.13, 33.63, 2.11),
	('Matangkuli', 'Mei', 2024, 143.48, 25.53, 1.34),
	('Langkahan', 'Juni', 2024, 223.11, 33.24, 0.82),
	('Baktiya', 'Juli', 2024, 54.56, 22.76, 1.7),
	('Matangkuli', 'Agustus', 2024, 102.72, 28.83, 1.54);

/*!40103 SET TIME_ZONE=IFNULL(@OLD_TIME_ZONE, 'system') */;
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IFNULL(@OLD_FOREIGN_KEY_CHECKS, 1) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40111 SET SQL_NOTES=IFNULL(@OLD_SQL_NOTES, 1) */;
