css = '''
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    width: 90%;
    margin: auto;
}

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
    margin-bottom: 1rem;    
    display: flex;
    max-width: 80%;
    word-wrap: break-word;
    font-size: 1.2rem;
}

.chat-message.user {
    background-color: #264653;
    color: #ecf0f1;
    align-self: flex-end;
    flex-direction: row-reverse;
    margin-left: auto;
}

.chat-message.bot {
    background-color: #34495E;
    color: #ecf0f1;
    align-self: flex-start;
    flex-direction: row;
    margin-right: auto;
}

.chat-message .avatar {
    width: 80px;
    margin-right: 1rem;
}

.chat-message.user .avatar {
    margin-left: 1rem;
    margin-right: 0;
}

.chat-message .avatar img {
    max-width: 80px;
    max-height: 80px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    padding: 0.75rem 1.5rem;
    color: inherit;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pravatar.cc/80?img=2">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

image_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message"><img src="data:image/png;base64,{{IMAGE_SRC}}" alt="Image"></div>
</div>
'''
