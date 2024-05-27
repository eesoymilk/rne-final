#include "DataServer.h"
#include <functional>
#include <cstring>

Message::Message()
{

}

Message::Message(const uint8_t *d, uint32_t length)
{
    data.resize(length);
    memcpy(data.data(), d, length);
}

Message::Message(const std::string &msg)
{
    data.reserve(msg.size());
    data.insert(data.end(), msg.cbegin(), msg.cend());
}

Action::Action(ActionType type, const std::string &route, websocketpp::connection_hdl from, const Message &msg, bool is_session)
    : type(type), route(route), from(from), msg(msg), is_session(is_session)
{
}

DataServer::DataServer(uint32_t port)
    : m_stop(false), m_cthread(std::bind(&DataServer::consumer, this))
{
    m_server.clear_access_channels(websocketpp::log::alevel::frame_header | websocketpp::log::alevel::frame_payload);
    m_server.set_open_handler(std::bind(&DataServer::on_open, this, std::placeholders::_1));
    m_server.set_close_handler(std::bind(&DataServer::on_close, this, std::placeholders::_1));
    m_server.set_message_handler(std::bind(&DataServer::on_message, this, std::placeholders::_1, std::placeholders::_2));
    m_server.set_http_handler(std::bind(&DataServer::on_http, this, std::placeholders::_1));
	m_server.init_asio();
	m_server.listen(port);
	m_server.start_accept();
}

void build_message(Message &msg, websocketpp::connection_hdl hdl, const std::string &payload)
{
    msg.data.resize(1+sizeof(intptr_t)+payload.size());
    msg.data[0] = static_cast<uint8_t>(SessionInfoOp::eMsg);
    *reinterpret_cast<intptr_t *>(&msg.data[1]) = reinterpret_cast<intptr_t>(hdl.lock().get());
    memcpy(msg.data.data()+1+sizeof(intptr_t), payload.data(), payload.size());
}

void DataServer::on_message(websocketpp::connection_hdl hdl, decltype(DataServer::m_server)::message_ptr msg)
{
    decltype(m_server)::connection_ptr con = m_server.get_con_from_hdl(hdl);
    con->get_resource().rfind('/');
    size_t pos = con->get_resource().rfind('/');

    std::string requestType = con->get_resource().substr(pos+1);
    std::string route = con->get_resource().substr(0, pos);

    if (requestType == "store") {
        auto &payload = msg->get_payload();
        auto &rdb = m_database[route];
        rdb.reserve(m_database[route].size() + payload.size());
        rdb.insert(rdb.end(), payload.cbegin(), payload.cend());
    } else if (requestType == "publish" || requestType == "chat") {
        auto &payload = msg->get_payload();
        std::lock_guard<std::mutex> guard(m_amutex);
        m_actions.emplace(ActionType::eMessage, route, hdl, payload);
        m_acond.notify_one();
    } else if (requestType == "session") {
        auto &payload = msg->get_payload();
        std::lock_guard<std::mutex> guard(m_amutex);
        m_actions.emplace(ActionType::eMessageNoSelf, route, hdl, payload);
        m_acond.notify_one();
    } else if (requestType == "session2") {
        auto &payload = msg->get_payload();
        switch(static_cast<SessionCommandOp>(payload[0])) {
            case SessionCommandOp::eBoardcast:
            {
                std::lock_guard<std::mutex> guard(m_amutex);
                Message msg;
                build_message(msg, hdl, payload.substr(1));
                m_actions.emplace(ActionType::eMessageNoSelf, route, hdl, msg);
                m_acond.notify_one();
                break;
            }
            
            case SessionCommandOp::eQueryParticipant:
            {
                auto &payload = msg->get_payload();
                std::lock_guard<std::mutex> guard(m_amutex);
                m_actions.emplace(ActionType::eQueryParticipant, route, hdl);
                m_acond.notify_one();
                break;
            }
        }

    }
}

void DataServer::on_http(websocketpp::connection_hdl hdl)
{
    decltype(m_server)::connection_ptr con = m_server.get_con_from_hdl(hdl);
    
    // Set status to 200 rather than the default error code
    con->set_status(websocketpp::http::status_code::ok);
    // Set body text to the HTML created above
    std::string ret;
    auto &rdb = m_database[con->get_resource()];
    ret.insert(ret.end(), rdb.cbegin(), rdb.cend());
    con->set_body(ret);
}

void DataServer::on_open(websocketpp::connection_hdl hdl)
{
    decltype(m_server)::connection_ptr con = m_server.get_con_from_hdl(hdl);
    size_t pos = con->get_resource().rfind('/');

    std::string requestType = con->get_resource().substr(pos+1);
    std::string route = con->get_resource().substr(0, pos);
    // eRead, eSubscribe, eChat, eSession
    if (requestType == "store") {
        // clear
        m_database[route] = {};
    }else if(requestType == "subscribe" || requestType == "chat"|| requestType == "session" || requestType == "session2") {
        std::lock_guard<std::mutex> guard(m_amutex);
        m_actions.emplace(ActionType::eSubscribe, route, hdl, Message{}, requestType == "session2");
        m_acond.notify_one();
    } else if(requestType == "publish") {
        // wait for message
    } else {
        // read
        route = con->get_resource();
        auto &rdb = m_database[route];
        std::cout<<rdb.size();
        m_server.send(hdl, rdb.data(), rdb.size(), websocketpp::frame::opcode::BINARY);
    }
}

void DataServer::on_close(websocketpp::connection_hdl hdl)
{
    decltype(m_server)::connection_ptr con = m_server.get_con_from_hdl(hdl);
    size_t pos = con->get_resource().rfind('/');
    std::string requestType = con->get_resource().substr(pos+1);
    std::string route = con->get_resource().substr(0, pos);
    if(requestType == "subscribe" || requestType == "chat"|| requestType == "session" || requestType == "session2") {
        // unsubscribe
        std::lock_guard<std::mutex> guard(m_amutex);
        m_actions.emplace(ActionType::eUnsubscribe, route, hdl, Message{}, requestType == "session2");
        m_acond.notify_one();
    }
}

template<class T>
static bool owner_equals(const std::shared_ptr<T> &lhs, const std::weak_ptr<T> &rhs) {
    return !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
}

void DataServer::consumer()
{
    while(!m_stop) {
        std::unique_lock<std::mutex> lock(m_amutex);
        // Wait for new object being pushed in and lock the queue.
        while(m_actions.empty()) {
            m_acond.wait(lock);
        }
        Action action = m_actions.front();
        m_actions.pop();
        lock.unlock();

        switch(action.type) {
            case ActionType::eStore:
            {
                std::lock_guard<std::mutex> guard(m_dbmutex);
                m_database[action.route] = action.msg.data;
                break;
            }
            case ActionType::eSubscribe:
            {
                auto &subs = m_subscribers[action.route];
                if(action.is_session) {
                    auto sender = action.from.lock();
                    std::vector<uint8_t> payload(1+sizeof(intptr_t));
                    payload[0] = static_cast<uint8_t>(SessionInfoOp::eEnter);
                    *reinterpret_cast<intptr_t *>(&payload[1]) = reinterpret_cast<intptr_t>(sender.get());
                    for(auto &sub: subs) {
                        if(!sub.expired())
                            m_server.send(sub, payload.data(), payload.size(), websocketpp::frame::opcode::BINARY);
                    }
                }
                subs.push_back(action.from);
                break;
            }
            case ActionType::eUnsubscribe:
            {
                auto &subs = m_subscribers[action.route];
                auto sender = action.from.lock();
                m_subscribers[action.route].remove_if([&sender](const websocketpp::connection_hdl &hdl) {
                    return owner_equals(sender, hdl);
                });
                if(action.is_session) {
                    auto sender = action.from.lock();
                    std::vector<uint8_t> payload(1+sizeof(intptr_t));
                    payload[0] = static_cast<uint8_t>(SessionInfoOp::eLeave);
                    *reinterpret_cast<intptr_t *>(&payload[1]) = reinterpret_cast<intptr_t>(sender.get());
                    for(auto &sub: subs) {
                        if(!sub.expired())
                            m_server.send(sub, payload.data(), payload.size(), websocketpp::frame::opcode::BINARY);
                    }
                }
                break;
            }
            case ActionType::eMessage:
            case ActionType::eMessageNoSelf:
            {
                auto &subs = m_subscribers[action.route];
                auto sender = action.from.lock();
                for(auto &sub: subs) {
                    if(action.type==ActionType::eMessageNoSelf && owner_equals(sender, sub))
                        continue;
                    if(!sub.expired())
                        m_server.send(sub, action.msg.data.data(), action.msg.data.size(), websocketpp::frame::opcode::BINARY);
                }
                // remove expired conn

                break;
            }
            case ActionType::eQueryParticipant:
            {
                auto &subs = m_subscribers[action.route];
                std::vector<uint8_t> payload;
                payload.resize(1+subs.size()*sizeof(intptr_t));
                payload[0] = static_cast<uint8_t>(SessionInfoOp::eQueryResult);
                int c = 0;
                for(auto &sub: subs) {
                    if(!sub.expired()) {
                        reinterpret_cast<intptr_t *>(payload.data()+1)[c] = reinterpret_cast<intptr_t>(sub.lock().get());
                        c++;
                    }
                }
                m_server.send(action.from, payload.data(), payload.size(), websocketpp::frame::opcode::BINARY);
            }
        }
    }
}

void DataServer::serve_forever()
{
    m_server.run();
    m_stop = true;
    m_cthread.join();
}

int main(int argc, char *argv[])
{
    DataServer server(9002);
    server.serve_forever();

    return 0;
}