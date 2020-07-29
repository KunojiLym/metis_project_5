from cryptography.fernet import Fernet
import psycopg2 as pg
import numpy as np

def generate_key():
    key = Fernet.generate_key()

    f = open("./data/sql_credentials.key", "w")
    f.write(key.decode())
    f.close()

def encrypt_credentials():
    f = open("./data/sql_credentials.key", "r")
    key = f.read().encode()

    username = input("Enter username: ").encode()
    password = input("Enter password: ").encode()

    cipher_suite = Fernet(key)
    e_username = cipher_suite.encrypt(username)
    e_password = cipher_suite.encrypt(password)

    f = open("./data/sql_credentials.hash", "w")
    f.write(e_username.decode() + "\n")
    f.write(e_password.decode())

    f.close()

def embedding_database():
    
    f = open("./data/sql_credentials.key", "r")
    key = f.read().encode()
    cipher_suite = Fernet(key)

    f = open("./data/sql_credentials.hash", "r")
    e_username = f.readline()
    e_password = f.readline()

    f.close()
    
    connection_args = {
        'host': 'localhost',  
        'dbname': 'word_embeddings',    
        'port': 5432,          
        'user': cipher_suite.decrypt(e_username.encode()).decode(),
        'password': cipher_suite.decrypt(e_password.encode()).decode()
    }

    return pg.connect(**connection_args)

def load_embeddings(file_name, table_name):
    """
    Assuumes the following format for embeddings:

    <token> <embed val 1> <embed val 2> .... <embed val n>
    <token> ...
    ...
    <token> ...
    """
    with open(file_name, encoding='utf8') as f:
        line = f.readline()
        line_tokens = line.split(' ')
        dimensions = len(line_tokens) - 1

        cursor = connection.cursor()

        table_creation_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                word_token      VARCHAR(255),
                embeddings      REAL[{dimensions}],
                PRIMARY KEY (word_token)
            );
        """
        cursor.execute(table_creation_sql)

        while line:
            line_tokens = line.split(' ')
            word_token = line_tokens[0]

            if len(word_token) > 255: 
                print('token length reached')
                break
            print(word_token, np.any([char != '$' for char in word_token]))
            
            word_token = decorate_string_for_sql(word_token)
            
            embeddings = line_tokens[1:]

            embeddings_sql = '{' + ', '.join(line_tokens[1:]) + '}'

            table_insertion_sql = f"""
                INSERT INTO {table_name} (word_token, embeddings) VALUES (
                    {word_token}, '{embeddings_sql}'
                )
            """
            cursor.execute(table_insertion_sql)

            line = f.readline()
        
        connection.commit()

def decorate_string_for_sql(string):

    if np.any([char == '$' for char in string]):
        string = "'" + string + "'"
    else:
        string = '$$' + string + '$$'

    return string

def call_embedding(word_token, table_name):

    cursor = connection.cursor()

    word_token = decorate_string_for_sql(word_token)

    call_embedding_sql = f"""
        SELECT embeddings FROM {table_name} 
        WHERE word_token = {word_token};
    """
    try:
        cursor.execute(call_embedding_sql)
        result = cursor.fetchall()
        return np.array(result[0][0])
    except:
        return np.zeros(300)

print('connecting to embedding database...', end='')
connection = embedding_database()
print('done')

#load_embeddings('./data/image_to_text/glove.840B.300d.txt', 'StanfordGlove')
