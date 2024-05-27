#ifndef _DATASERVER_H_
#define _DATASERVER_H_

#include <iostream>
#include <cstdint>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <map>
#include <vector>
#include <list>
#include <mutex>
#include <future>
#include <thread>

// ws://localhost:port/data_route../.../ <- Read once, don't subscribe
// ws:://localhost:port/data_route../.../store
// ws://localhost:port/data_route../.../subscribe <- subscribe this data
// ws://localhost:port/data_route../.../publish <- publish data
// ws://localhost:port/data_route../.../chat <- p2p or many to many
// ws://localhost:port/data_route../.../session <- p2p or many to many communication, (don't send back to self)
// ws://localhost:port/data_route../.../session2 <- Session 2.0

enum class RequestType {
    eRead,
    eStore,
    eSubscribe,
    ePublish,
    eChat,
    eSession,
    eSession2
};

enum class SessionInfoOp {
    eMsg=0xAA,
    eQueryResult=0xAB,
    eEnter=0xBB,
    eLeave=0xCC
};

enum class SessionCommandOp {
    eBoardcast,
    // (not implemented)
    eName=0x01,
    eQueryParticipant,
    // send to someone(not implemented)
    eSlient,
    // admin command
    eAdmin=0x80,
    // Kick someone (not implemented)
    eAdminKick=0x81
};

enum class ActionType {
    eStore,
    eSubscribe,
    eUnsubscribe,
    eMessage,
    // Message never sent back to sender
    eMessageNoSelf,
    eQueryParticipant,
    eGarbageCollection
};

struct Message {
    std::vector<uint8_t> data;
    Message();
    Message(const std::string &);
    Message(const uint8_t *data, uint32_t length);
};

struct Action {
    ActionType type;
    std::string route;
    websocketpp::connection_hdl from;
    Message msg;
    bool is_session;
    Action(ActionType type, const std::string &route, websocketpp::connection_hdl from, const Message &msg=Message(), bool is_session=false);
};

class DataServer {
public:
    DataServer(uint32_t port);

    void serve_forever();
    void subscribe(const std::string &route);
    void publish(const std::string &route, const uint8_t *data, uint32_t length);


private:
    websocketpp::server<websocketpp::config::asio> m_server;

    void on_open(websocketpp::connection_hdl hdl);
    void on_close(websocketpp::connection_hdl hdl);
    void on_message(websocketpp::connection_hdl hdl, decltype(DataServer::m_server)::message_ptr msg);
    void on_http(websocketpp::connection_hdl hdl);
    void consumer();


    bool m_stop;

    std::mutex m_dbmutex;
    std::map<std::string, std::vector<uint8_t>> m_database;
    std::map<std::string, std::list<websocketpp::connection_hdl>> m_subscribers;
    std::mutex m_amutex;
    std::condition_variable m_acond;
    std::queue<Action> m_actions;

    std::thread m_cthread;

};

#endif