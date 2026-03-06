CREATE TABLE walmart_sales (
    date DATE,
    time_of_purchase TIME,
    branch VARCHAR(5),
    city VARCHAR(50),
    region VARCHAR(50),
    product_line VARCHAR(100),
    customer_type VARCHAR(20),
    sales_channel VARCHAR(20),
    quantity INT,
    unit_price FLOAT,
    discount FLOAT,
    revenue FLOAT,
    profit FLOAT,
    payment_method VARCHAR(50),
    customer_rating FLOAT
);

SELECT * FROM walmart_sales
LIMIT 10;

SELECT SUM(revenue) AS total_revenue
FROM walmart_sales;

SELECT SUM(profit) AS total_profit
FROM walmart_sales;

SELECT product_line,
SUM(revenue) AS total_sales
FROM walmart_sales
GROUP BY product_line
ORDER BY total_sales DESC;

SELECT branch,
SUM(revenue) AS revenue,
SUM(profit) AS profit
FROM walmart_sales
GROUP BY branch;

SELECT
EXTRACT(HOUR FROM time_of_purchase) AS hour,
SUM(revenue) AS total_sales
FROM walmart_sales
GROUP BY hour
ORDER BY total_sales DESC;

SELECT payment_method,
COUNT(*) AS transactions
FROM walmart_sales
GROUP BY payment_method
ORDER BY transactions DESC;

SELECT
DATE_TRUNC('month', date) AS month,
SUM(revenue) AS total_sales
FROM walmart_sales
GROUP BY month
ORDER BY month;

SELECT customer_type,
AVG(revenue) AS avg_spending
FROM walmart_sales
GROUP BY customer_type;

SELECT product_line,
SUM(profit)/SUM(revenue)*100 AS profit_margin
FROM walmart_sales
GROUP BY product_line
ORDER BY profit_margin DESC;

SELECT branch,
product_line,
SUM(quantity) AS units_sold
FROM walmart_sales
GROUP BY branch, product_line
ORDER BY branch;