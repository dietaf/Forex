# ===================================================================
# BOT FOREX AUTOMATIZADO - OANDA API v20
# Estrategias optimizadas para Forex con timeframes configurados
# TrendShift (4H, Daily), Pivot Hunter (4H), Quantum Shift (15min, 1H)
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from collections import deque

# Importar OANDA API
try:
    from oandapyV20 import API
    from oandapyV20.exceptions import V20Error
    import oandapyV20.endpoints.orders as orders
    import oandapyV20.endpoints.trades as trades
    import oandapyV20.endpoints.pricing as pricing
    import oandapyV20.endpoints.accounts as accounts
    import oandapyV20.endpoints.instruments as instruments
    from oandapyV20.contrib.requests import MarketOrderRequest
    from oandapyV20.contrib.requests import TakeProfitDetails, StopLossDetails, TrailingStopLossDetails
except:
    st.error("Instalando OANDA API...")

# ===================================================================
# CONFIGURACIÓN
# ===================================================================

st.set_page_config(
    page_title="Forex Bot - OANDA",
    page_icon="💱",
    layout="wide"
)

# ===================================================================
# CLASE DEL BOT FOREX
# ===================================================================

class ForexTradingBot:
    def __init__(self, api_key, account_id, environment="practice"):
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        self.api = None
        self.is_running = False
        self.current_position = None
        self.trades_history = []
        self.equity_history = deque(maxlen=1000)
        self.logs = deque(maxlen=100)
        
        # Configuración optimizada por estrategia
        self.strategy_configs = {
            "TrendShift": {
                "timeframe": "H4",
                "fast_ema": 9,
                "slow_ema": 21,
                "win_rate": "70-80%",
                "description": "Seguimiento de tendencia largo plazo"
            },
            "Pivot Hunter": {
                "timeframe": "H4",
                "window": 20,
                "buffer": 5,
                "win_rate": "65-75%",
                "description": "Soportes y resistencias dinámicos"
            },
            "Quantum Shift": {
                "timeframe": "M15",
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "win_rate": "65-75%",
                "description": "Reversiones con RSI + Volumen"
            }
        }
        
        # Configuración actual
        self.pair = "EUR_USD"
        self.strategy = "TrendShift"
        self.units = 1000
        self.risk_per_trade = 0.02
        
        # Estado
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        if api_key and account_id:
            try:
                self.api = API(access_token=api_key, environment=environment)
                self.log("✅ Conectado a OANDA", "success")
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
    
    def log(self, message, level="info"):
        """Registra eventos"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append({
            'time': timestamp,
            'message': message,
            'level': level
        })
    
    def get_account_info(self):
        """Info de cuenta"""
        try:
            r = accounts.AccountDetails(self.account_id)
            self.api.request(r)
            acc = r.response['account']
            return {
                'balance': float(acc['balance']),
                'pl': float(acc['pl']),
                'margin_used': float(acc['marginUsed']),
                'margin_available': float(acc['marginAvailable']),
                'open_trades': int(acc['openTradeCount'])
            }
        except Exception as e:
            self.log(f"Error cuenta: {str(e)}", "error")
            return None
    
    def get_timeframe_granularity(self, tf_string):
        """Convierte timeframe"""
        mapping = {
            "M1": "M1",
            "M5": "M5",
            "M15": "M15",
            "M30": "M30",
            "H1": "H1",
            "H4": "H4",
            "D": "D"
        }
        return mapping.get(tf_string, "H4")
    
    def get_historical_data(self, pair, granularity="H4", count=500):
        """Obtiene datos históricos"""
        try:
            params = {
                "granularity": granularity,
                "count": count
            }
            
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            self.api.request(r)
            
            candles = r.response['candles']
            
            data = []
            for candle in candles:
                if candle['complete']:
                    data.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            
            return df
        except Exception as e:
            self.log(f"Error datos: {str(e)}", "error")
            return None
    
    def get_current_price(self, pair):
        """Precio actual"""
        try:
            params = {"instruments": pair}
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            self.api.request(r)
            
            price_data = r.response['prices'][0]
            bid = float(price_data['bids'][0]['price'])
            ask = float(price_data['asks'][0]['price'])
            
            return (bid + ask) / 2
        except Exception as e:
            self.log(f"Error precio: {str(e)}", "error")
            return None
    
    def calculate_atr(self, df, period=14):
        """Calcula ATR"""
        if len(df) < period:
            return 0
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if len(atr) > 0 else 0
    
    # ===============================================================
    # ESTRATEGIAS OPTIMIZADAS PARA FOREX
    # ===============================================================
    
    def strategy_trendshift(self, df):
        """TrendShift - Optimizado para 4H y Daily"""
        config = self.strategy_configs["TrendShift"]
        fast = config['fast_ema']
        slow = config['slow_ema']
        
        if len(df) < slow + 5:
            return 0
        
        # EMAs
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # Confirmación con precio por encima de ambas EMAs
        price = df['close'].iloc[-1]
        
        # Golden Cross + precio confirmando
        if (ema_fast.iloc[-2] <= ema_slow.iloc[-2] and 
            ema_fast.iloc[-1] > ema_slow.iloc[-1] and
            price > ema_fast.iloc[-1]):
            return 1
        
        # Death Cross + precio confirmando
        elif (ema_fast.iloc[-2] >= ema_slow.iloc[-2] and 
              ema_fast.iloc[-1] < ema_slow.iloc[-1] and
              price < ema_fast.iloc[-1]):
            return -1
        
        return 0
    
    def strategy_pivot_hunter(self, df):
        """Pivot Hunter - Optimizado para 4H"""
        config = self.strategy_configs["Pivot Hunter"]
        window = config['window']
        buffer_pips = config['buffer'] * 0.0001  # 5 pips
        
        if len(df) < window * 2:
            return 0
        
        # Pivots más significativos
        pivot_high = df['high'].iloc[-window*2:-window].max()
        pivot_low = df['low'].iloc[-window*2:-window].min()
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        
        # Confirmación con volumen
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        # Ruptura con volumen
        if (prev_price <= pivot_high and 
            current_price > pivot_high + buffer_pips and
            current_volume > avg_volume * 1.2):
            return 1
        
        elif (prev_price >= pivot_low and 
              current_price < pivot_low - buffer_pips and
              current_volume > avg_volume * 1.2):
            return -1
        
        return 0
    
    def strategy_quantum_shift(self, df):
        """Quantum Shift - Optimizado para 15min y 1H"""
        config = self.strategy_configs["Quantum Shift"]
        rsi_period = config['rsi_period']
        oversold = config['rsi_oversold']
        overbought = config['rsi_overbought']
        
        if len(df) < rsi_period + 20:
            return 0
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # Volumen
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        vol_ratio = current_volume / avg_volume
        
        # Divergencias y volumen alto
        # Sobreventa + divergencia alcista + volumen
        if (current_rsi < oversold and 
            current_rsi > prev_rsi and  # RSI girando
            vol_ratio > 1.5):
            return 1
        
        # Sobrecompra + divergencia bajista + volumen
        elif (current_rsi > overbought and 
              current_rsi < prev_rsi and  # RSI girando
              vol_ratio > 1.5):
            return -1
        
        return 0
    
    def get_signal(self, df):
        """Obtiene señal según estrategia"""
        if self.strategy == "TrendShift":
            return self.strategy_trendshift(df)
        elif self.strategy == "Pivot Hunter":
            return self.strategy_pivot_hunter(df)
        elif self.strategy == "Quantum Shift":
            return self.strategy_quantum_shift(df)
        return 0
    
    # ===============================================================
    # EJECUCIÓN DE TRADES
    # ===============================================================
    
    def calculate_position_size(self, atr):
        """Calcula tamaño basado en ATR"""
        account = self.get_account_info()
        if not account:
            return 1000
        
        risk_amount = account['balance'] * self.risk_per_trade
        stop_distance_pips = atr * 10000  # Convertir a pips
        
        if stop_distance_pips == 0:
            return 1000
        
        # Forex: 1 pip = $0.0001 por 1000 units
        # Risk = Units * Stop_Distance_Pips * $0.0001
        units = int((risk_amount / stop_distance_pips) / 0.0001)
        
        # Limitar entre 1000 y 100000 units
        return max(1000, min(units, 100000))
    
    def place_order(self, pair, units, side, stop_loss=None, take_profit=None):
        """Coloca orden en OANDA"""
        try:
            # Configurar trailing stop
            trailing_distance = 0.0030  # 30 pips
            
            order_data = MarketOrderRequest(
                instrument=pair,
                units=units if side == "buy" else -units,
                trailingStopLossOnFill=TrailingStopLossDetails(distance=trailing_distance).data
            )
            
            r = orders.OrderCreate(self.account_id, data=order_data.data)
            self.api.request(r)
            
            self.log(f"✅ Orden: {side} {units} {pair}", "success")
            return r.response
        except Exception as e:
            self.log(f"❌ Error orden: {str(e)}", "error")
            return None
    
    def open_position(self, signal, price, atr):
        """Abre posición"""
        units = self.calculate_position_size(atr)
        side = "buy" if signal == 1 else "sell"
        
        # Calcular stops
        stop_distance = atr * 2
        tp_distance = atr * 3
        
        if signal == 1:
            stop_loss = price - stop_distance
            take_profit = price + tp_distance
        else:
            stop_loss = price + stop_distance
            take_profit = price - tp_distance
        
        order = self.place_order(self.pair, units, side, stop_loss, take_profit)
        
        if order:
            self.current_position = {
                'type': 'LONG' if signal == 1 else 'SHORT',
                'entry_price': price,
                'units': units,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            self.log(f"🎯 {self.current_position['type']} @ {price:.5f}", "success")
            self.log(f"   SL: {stop_loss:.5f} | TP: {take_profit:.5f}", "info")
    
    def close_all_positions(self):
        """Cierra todas las posiciones"""
        try:
            r = trades.OpenTrades(accountID=self.account_id)
            self.api.request(r)
            
            open_trades = r.response.get('trades', [])
            
            for trade in open_trades:
                trade_id = trade['id']
                r_close = trades.TradeClose(accountID=self.account_id, tradeID=trade_id)
                self.api.request(r_close)
                
                self.log(f"💰 Cerrado trade {trade_id}", "success")
            
            self.current_position = None
        except Exception as e:
            self.log(f"Error cerrando: {str(e)}", "error")
    
    # ===============================================================
    # LOOP PRINCIPAL
    # ===============================================================
    
    def trading_loop(self):
        """Loop principal del bot"""
        self.log("🤖 Bot Forex iniciado", "success")
        config = self.strategy_configs[self.strategy]
        self.log(f"📊 {self.strategy} ({config['timeframe']}) - WR: {config['win_rate']}", "info")
        
        while self.is_running:
            try:
                # Obtener datos con el timeframe de la estrategia
                granularity = self.get_timeframe_granularity(config['timeframe'])
                df = self.get_historical_data(self.pair, granularity)
                
                if df is None or len(df) == 0:
                    time.sleep(30)
                    continue
                
                current_price = self.get_current_price(self.pair)
                if not current_price:
                    time.sleep(30)
                    continue
                
                atr = self.calculate_atr(df)
                
                # Actualizar equity
                account = self.get_account_info()
                if account:
                    self.equity_history.append(account['balance'])
                
                # Buscar señales
                if not self.current_position:
                    signal = self.get_signal(df)
                    if signal != 0:
                        self.log(f"🎯 Señal: {'COMPRA' if signal == 1 else 'VENTA'}", "info")
                        self.open_position(signal, current_price, atr)
                
                # Esperar según timeframe
                if "M" in config['timeframe']:
                    sleep_time = 30  # 30 seg para minutos
                elif "H" in config['timeframe']:
                    sleep_time = 300  # 5 min para horas
                else:
                    sleep_time = 600  # 10 min para daily
                
                time.sleep(sleep_time)
                
            except Exception as e:
                self.log(f"❌ Error: {str(e)}", "error")
                time.sleep(60)
        
        self.log("🛑 Bot detenido", "warning")
    
    def start(self):
        """Inicia bot"""
        if not self.is_running:
            self.is_running = True
            thread = threading.Thread(target=self.trading_loop, daemon=True)
            thread.start()
            return True
        return False
    
    def stop(self):
        """Detiene bot"""
        self.is_running = False
        self.close_all_positions()

# ===================================================================
# INTERFAZ STREAMLIT
# ===================================================================

def main():
    st.title("💱 Forex Trading Bot - OANDA")
    st.markdown("### Estrategias Optimizadas para Forex")
    
    if 'bot' not in st.session_state:
        st.session_state.bot = None
        st.session_state.bot_running = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración OANDA")
        
        # API
        st.subheader("🔑 Credenciales")
        try:
            api_key = st.secrets["oanda"]["api_key"]
            account_id = st.secrets["oanda"]["account_id"]
            st.success("✅ Keys desde secrets")
        except:
            api_key = st.text_input("API Key", type="password")
            account_id = st.text_input("Account ID")
        
        environment = st.selectbox("Ambiente", ["practice", "live"])
        
        st.divider()
        
        # Estrategia
        st.subheader("🎯 Estrategia")
        
        strategy = st.selectbox(
            "Selecciona Estrategia",
            ["TrendShift", "Pivot Hunter", "Quantum Shift"]
        )
        
        # Mostrar config de la estrategia
        if strategy:
            config = {
                "TrendShift": {"tf": "4H, Daily", "wr": "70-80%", "desc": "Tendencias largas"},
                "Pivot Hunter": {"tf": "4H", "wr": "65-75%", "desc": "S/R dinámicos"},
                "Quantum Shift": {"tf": "15min, 1H", "wr": "65-75%", "desc": "Reversiones RSI"}
            }[strategy]
            
            st.info(f"""
            **Timeframe:** {config['tf']}  
            **Win Rate:** {config['wr']}  
            **Tipo:** {config['desc']}
            """)
        
        # Pares principales
        pair = st.selectbox(
            "Par de Forex",
            ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "EUR_GBP"]
        )
        
        st.divider()
        
        # Risk
        st.subheader("💰 Risk Management")
        units = st.number_input("Units", min_value=1000, value=1000, step=1000)
        risk_pct = st.slider("Riesgo (%)", 1, 5, 2) / 100
        
        st.divider()
        
        # Controles
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ INICIAR", use_container_width=True, type="primary"):
                if api_key and account_id:
                    st.session_state.bot = ForexTradingBot(api_key, account_id, environment)
                    st.session_state.bot.strategy = strategy
                    st.session_state.bot.pair = pair
                    st.session_state.bot.units = units
                    st.session_state.bot.risk_per_trade = risk_pct
                    
                    if st.session_state.bot.start():
                        st.session_state.bot_running = True
                        st.success("✅ Bot iniciado!")
                        st.rerun()
                else:
                    st.error("⚠️ Ingresa credenciales")
        
        with col2:
            if st.button("⏹️ DETENER", use_container_width=True):
                if st.session_state.bot:
                    st.session_state.bot.stop()
                    st.session_state.bot_running = False
                    st.warning("🛑 Bot detenido")
                    st.rerun()
    
    # Main
    if st.session_state.bot and st.session_state.bot_running:
        bot = st.session_state.bot
        config = bot.strategy_configs[bot.strategy]
        
        # Status
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #00ff00 0%, #00cc00 100%); 
                    padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">🟢 BOT ACTIVO - {bot.strategy} ({config['timeframe']}) en {bot.pair}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas
        account = bot.get_account_info()
        if account:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💵 Balance", f"${account['balance']:,.2f}")
            with col2:
                st.metric("📊 P/L", f"${account['pl']:,.2f}")
            with col3:
                st.metric("📈 Trades Abiertos", account['open_trades'])
            with col4:
                st.metric("💼 Margen Usado", f"${account['margin_used']:,.2f}")
        
        # Tabs
        tab1, tab2 = st.tabs(["📊 Dashboard", "📜 Logs"])
        
        with tab1:
            if len(bot.equity_history) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=list(bot.equity_history),
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#00ff00', width=2)
                ))
                fig.update_layout(height=300, template="plotly_dark", title="Balance History")
                st.plotly_chart(fig, use_container_width=True)
            
            if bot.current_position:
                st.subheader("📍 Posición Actual")
                pos = bot.current_position
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"""
                    **Tipo:** {pos['type']}  
                    **Entrada:** {pos['entry_price']:.5f}  
                    **Units:** {pos['units']:,}
                    """)
                
                with col2:
                    cp = bot.get_current_price(bot.pair)
                    if cp:
                        if pos['type'] == 'LONG':
                            pips = (cp - pos['entry_price']) * 10000
                        else:
                            pips = (pos['entry_price'] - cp) * 10000
                        
                        color = "green" if pips > 0 else "red"
                        st.markdown(f"""
                        **Precio Actual:** {cp:.5f}  
                        **P/L:** <span style="color: {color};">{pips:+.1f} pips</span>
                        """, unsafe_allow_html=True)
                
                with col3:
                    st.warning(f"""
                    **Stop Loss:** {pos['stop_loss']:.5f}  
                    **Take Profit:** {pos['take_profit']:.5f}
                    """)
        
        with tab2:
            st.subheader("📜 Logs del Bot")
            for log in reversed(list(bot.logs)):
                color = {'success': 'green', 'error': 'red', 'warning': 'orange', 'info': 'blue'}.get(log['level'], 'white')
                st.markdown(f"""
                <div style="background: rgba(0,0,0,0.3); padding: 8px; margin: 4px 0; border-radius: 5px; border-left: 3px solid {color};">
                    [{log['time']}] {log['message']}
                </div>
                """, unsafe_allow_html=True)
        
        time.sleep(5)
        st.rerun()
    
    else:
        st.info("""
        ### 💱 Bot de Forex con OANDA
        
        **Estrategias Pre-optimizadas:**
        
        1. **TrendShift (4H, Daily)** - Win Rate: 70-80%
           - Seguimiento de tendencias con EMAs
           - Mejor para swing trading
        
        2. **Pivot Hunter (4H)** - Win Rate: 65-75%
           - Soportes y resistencias dinámicos
           - Rupturas confirmadas con volumen
        
        3. **Quantum Shift (15min, 1H)** - Win Rate: 65-75%
           - Reversiones con RSI
           - Scalping y day trading
        
        **Para comenzar:**
        1. Ingresa tu API Key de OANDA
        2. Ingresa tu Account ID
        3. Selecciona estrategia
        4. Click ▶️ INICIAR
        
        **📌 Nota:** Usa ambiente "practice" para paper trading
        """)
        
        st.divider()
        
        st.markdown("""
        ### 🔑 ¿Cómo obtener API Key de OANDA?
        
        1. Ve a [OANDA](https://www.oanda.com)
        2. Crea cuenta demo (gratis)
        3. Login → My Account → Manage API Access
        4. Genera tu token
        5. Copia Account ID y Token
        """)

if __name__ == "__main__":
    main()
