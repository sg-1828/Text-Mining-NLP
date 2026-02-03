# Loading Necessary Libraries

#install.packages(c("mongolite","tm", "reshape2", "topicmodels", "tidytext", "stringr", "igraph"))
library(mongolite)
library(dplyr)
library(tidytext)
library(tidyverse)
library(stringr)
library(ggplot2)
library(igraph)
library(ggraph)
library(tm)
library(topicmodels)
library(reshape2)

# Setting up MongoDb Connection

# MongoDB connection string setup
connection_string <- 'mongodb+srv://saichand1422:12345@cluster0.fxih6yq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
airbnb_collection <- mongo(collection="listingsAndReviews", db="sample_airbnb", url=connection_string)

# Downloading the data
airbnb_all <- airbnb_collection$find()


# Data Wrangling and Exploring

# Convert MongoDB data to a dataframe
airbnb_df <- as.data.frame(airbnb_all)

# Overview of the data structure
print(colnames(airbnb_df))


# Text Analysis
# Tokenization and frequency Analysis

# Tokenization of the 'description' column
token_list <- airbnb_df %>%
  unnest_tokens(word, description)  # Adjust column name if different

# Word Frequency Analysis, excluding stop words
frequencies_token_nostop <- token_list %>%
  anti_join(stop_words) %>%
  count(word, sort = TRUE)

# Top 15 Words Visualization
freq_hist <- frequencies_token_nostop %>%
  top_n(15) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(x = word, y = n)) +
  geom_col() +
  coord_flip() +
  labs(y = "Frequency", title = "Top 15 Words in Airbnb Description of houses")

print(freq_hist)


# Bigram Analysis

# Bigrams, avoiding stop words, and plotting the network graph
airbnb_df_bigrams <- airbnb_df %>%
  unnest_tokens(bigram, description, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word & !word2 %in% stop_words$word) %>%
  count(word1, word2, sort = TRUE) %>%
  top_n(15, n)

bigram_graph <- graph_from_data_frame(airbnb_df_bigrams)
ggraph(bigram_graph, layout = "fr") +
  geom_edge_link(aes(width = n), edge_colour = "blue", edge_alpha = 0.5) +
  geom_node_point(color = "darkred", size = 5) +
  geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
  labs(title = "Top 15 Most Frequent Bigrams", subtitle = "Filtered to exclude common stopwords")


# Sentiment Analysis

# Sentiment Analysis using Bing lexicon
sentiments <- get_sentiments("bing")
sentiment_analysis <- airbnb_df %>%
  unnest_tokens(word, description) %>%
  inner_join(sentiments) %>%
  count(word, sentiment, sort = TRUE) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment_score = positive - negative)

print(sentiment_analysis)

# Selecting the top 15 words based on absolute sentiment score
top_words <- sentiment_analysis %>%
  mutate(abs_score = abs(sentiment_score)) %>%
  arrange(desc(abs_score)) %>%
  slice_head(n = 15)

# Plotting
ggplot(top_words, aes(x = reorder(word, sentiment_score), y = sentiment_score, fill = sentiment_score > 0)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(
    name = "Sentiment Type",
    values = c("TRUE" = "blue", "FALSE" = "red"),  # Correct assignment inside c()
    labels = c("Positive", "Negative")
  ) +
  labs(
    title = "Top 15 Words by Sentiment Score",
    x = "Word",
    y = "Sentiment Score"
  ) +
  theme_minimal()


# TF-IDF Analysis

# Calculating TF-IDF using 'listing_url' as the document identifier
tf_idf <- airbnb_df %>%
  unnest_tokens(word, description) %>%
  anti_join(stop_words) %>%
  count(listing_url, word, sort = TRUE) %>%
  bind_tf_idf(word, listing_url, n)

# Visualization of the highest TF-IDF scores
top_tfidf <- tf_idf %>%
  arrange(desc(tf_idf)) %>%
  top_n(15)

ggplot(top_tfidf, aes(x = reorder(word, tf_idf), y = tf_idf)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top TF-IDF Scores in Airbnb Descriptions", x = "Terms", y = "TF-IDF")


# LDA for topic Modelling


# Preparing the document-term matrix using 'listing_url' as the document identifier
airbnb_df_filtered <- airbnb_df %>%
  unnest_tokens(word, description) %>%
  anti_join(stop_words, by = "word")

dtm <- airbnb_df_filtered %>%
  count(listing_url, word) %>%
  cast_dtm(listing_url, word, n)

# Fitting the LDA model
lda_model <- LDA(dtm, k = 5)  # k is the number of topics

# Extract the top terms from the LDA model for visualization
topics <- tidy(lda_model, matrix = "beta")

# Filter for the highest probability terms in each topic
top_terms <- topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Plotting
ggplot(top_terms, aes(x = reorder(term, -beta), y = beta, fill = as.factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  labs(title = "Top Terms in Each Topic from LDA Model", x = "Terms", y = "Beta")



library(shiny)

# Define UI for application that draws various plots
ui <- fluidPage(
  # Application title
  titlePanel("Airbnb Text Analysis Dashboard"),
  
  # Sidebar with a slider input for number of words
  sidebarLayout(
    sidebarPanel(
      sliderInput("nWords",
                  "Number of Words:",
                  min = 1,
                  max = 50,
                  value = 15)
    ),
    
    # Show a plot of the generated distribution and text outputs
    mainPanel(
      tabsetPanel(
        tabPanel("Frequency", plotOutput("freqPlot")),
        tabPanel("Sentiment", plotOutput("sentPlot")),
        tabPanel("Bigrams", plotOutput("bigramPlot")),
        tabPanel("TF-IDF", plotOutput("tfidfPlot")),
        tabPanel("LDA", plotOutput("ldaPlot"))
      )
    )
  )
)

# Define server logic required to draw the plots
server <- function(input, output, session) {
  
  # Frequency Plot
  output$freqPlot <- renderPlot({
    top_n_words <- frequencies_token_nostop %>%
      top_n(input$nWords) %>%
      mutate(word = reorder(word, n))
    ggplot(top_n_words, aes(x = word, y = n)) +
      geom_col() +
      coord_flip() +
      labs(y = "Frequency", title = "Word Frequencies")
  })
  
  # Sentiment Plot
  output$sentPlot <- renderPlot({
    top_n_words <- sentiment_analysis %>%
      top_n(input$nWords, abs(sentiment_score)) %>%
      mutate(word = reorder(word, sentiment_score))
    ggplot(top_n_words, aes(x = word, y = sentiment_score, fill = sentiment_score > 0)) +
      geom_col() +
      coord_flip() +
      scale_fill_manual(values = c("TRUE" = "blue", "FALSE" = "red")) +
      labs(y = "Sentiment Score", title = "Sentiment Scores")
  })
  
  # Bigrams Plot
  output$bigramPlot <- renderPlot({
    bigram_data <- airbnb_df_bigrams %>%
      top_n(input$nWords, n)
    
    set.seed(123)
    bigram_graph <- graph_from_data_frame(bigram_data)
    ggraph(bigram_graph, layout = "fr") +
      geom_edge_link(aes(width = n), edge_colour = "blue", edge_alpha = 0.5) +
      geom_node_point(color = "darkred", size = 5) +
      geom_node_text(aes(label = name), vjust = 1, hjust = 1) +
      labs(title = "Top Bigrams")
  })
  
  # TF-IDF Plot
  output$tfidfPlot <- renderPlot({
    top_tfidf_words <- top_tfidf %>%
      top_n(input$nWords)
    ggplot(top_tfidf_words, aes(x = reorder(word, tf_idf), y = tf_idf)) +
      geom_col() +
      coord_flip() +
      labs(y = "TF-IDF Score", title = "TF-IDF Scores")
  })
  
  # LDA Plot
  output$ldaPlot <- renderPlot({
    top_terms <- topics %>%
      group_by(topic) %>%
      top_n(10, beta) %>%
      ungroup() %>%
      arrange(topic, -beta)
    ggplot(top_terms, aes(x = reorder(term, -beta), y = beta, fill = as.factor(topic))) +
      geom_col(show.legend = FALSE) +
      facet_wrap(~ topic, scales = "free_y") +
      coord_flip() +
      labs(title = "Top Terms in Each Topic from LDA Model", x = "Terms", y = "Beta")
  })
}
# Run the application 
shinyApp(ui = ui, server = server)


library(leaflet)
library(TD)

str(airbnb_df$address)
str(airbnb_df$review_scores)


# Define UI
ui <- fluidPage(
  titlePanel("Airbnb Listings Map"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("country", "Country:",
                  choices = c("All", unique(airbnb_all$address$country))),
      selectInput("propertyType", "Property Type:",
                  choices = c("All", unique(airbnb_all$property_type))),
      sliderInput("priceRange", "Price Range:",
                  min = 0, max = 4000, value = c(0, 4000)),
      sliderInput("bedroomsInput", "Number of Bedrooms:", 
                  min = 0, max = max(airbnb_all$bedrooms, na.rm = TRUE), 
                  value = c(0, max(airbnb_all$bedrooms, na.rm = TRUE))),
      sliderInput("bathroomsInput", "Number of Bathrooms:", 
                  min = 0, max = max(airbnb_all$bathrooms, na.rm = TRUE), 
                  value = c(0, max(airbnb_all$bathrooms, na.rm = TRUE)))
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Map", leafletOutput("map")),
        tabPanel("Top 10 Rentals", DTOutput("topRentals")),
        tabPanel("Bottom 10 Rentals", DTOutput("bottomRentals"))
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  
  # Convert coordinates to separate lon and lat columns
  airbnb_all <- airbnb_all %>%
    mutate(
      lon = map_dbl(address$location$coordinates, 1),
      lat = map_dbl(address$location$coordinates, 2),
      review_scores_rating = as.numeric(review_scores$review_scores_rating),
      bedrooms = as.numeric(bedrooms),
      bathrooms = as.numeric(bathrooms)
    )
  
  # Reactive expression for filtered data
  filteredData <- reactive({
    data <- airbnb_all
    
    if (input$country != "All") {
      data <- data %>% filter(address$country == input$country)
    }
    
    if (input$propertyType != "All") {
      data <- data %>% filter(property_type == input$propertyType)
    }
    
    data <- data %>%
      filter(price >= input$priceRange[1] & price <= input$priceRange[2],
             bedrooms >= input$bedroomsInput[1] & bedrooms <= input$bedroomsInput[2],
             bathrooms >= input$bathroomsInput[1] & bathrooms <= input$bathroomsInput[2])
    
    data
  })
  
  # Render the Leaflet map
  output$map <- renderLeaflet({
    leaflet(data = filteredData()) %>%
      addTiles() %>%
      addMarkers(~lon, ~lat, popup = ~paste(property_type, price))
  })
  
  # Render top 10 rentals data table
  output$topRentals <- renderDT({
    filteredData() %>%
      arrange(desc(review_scores_rating)) %>%
      head(10)
  }, options = list(pageLength = 10, scrollX = TRUE))
  
  # Render bottom 10 rentals data table
  output$bottomRentals <- renderDT({
    filteredData() %>%
      arrange(review_scores_rating) %>%
      head(10)
  }, options = list(pageLength = 10, scrollX = TRUE))
}

# Run the Shiny app
shinyApp(ui = ui, server = server)
