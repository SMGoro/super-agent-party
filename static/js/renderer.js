// 在页面加载完成后添加 content-loaded 类
document.addEventListener('DOMContentLoaded', function() {
  // 设置一个短暂的延迟，确保所有资源都已加载
  setTimeout(function() {
    document.body.classList.add('content-loaded');
  }, 100);
});

// 创建Vue应用
const app = Vue.createApp({
  data() {
    return vue_data
  },
  // 在组件销毁时清除定时器
  beforeDestroy() {
    if (this.behaviorTimeTimer)   clearInterval(this.behaviorTimeTimer)
    if (this.behaviorNoInputTimer) clearInterval(this.behaviorNoInputTimer)
    clearInterval(this.behaviorCycleTimer);
    this.cycleTimers.forEach(timer => {
      if (timer) clearInterval(timer);
    });
    if (this.statusInterval) {
      clearInterval(this.statusInterval);
    }
    window.removeEventListener('keydown', this.handleKeyDown);
    window.removeEventListener('keyup', this.handleKeyUp);
    window.removeEventListener('resize', this.checkMobile);
    this.shouldReconnectWs = false; // 设置标志位
    this.stopDanmuProcessor(); // 停止弹幕处理器
    this.disconnectWebSocket();
  },
  async mounted() {
    this.checkMobile();
    window.addEventListener('resize', this.handleResize);
    if (isElectron) {
      this.checkServerPort();
    }
    window.addEventListener('keydown', this.handleKeyDown)
    window.addEventListener('keyup', this.handleKeyUp)
    window.addEventListener('resize', this.checkMobile);

    if (isElectron) {
      this.isMac = window.electron.isMac;
      this.isWindows = window.electron.isWindows;
    }
    this.initWebSocket();
    this.highlightCode();
    this.initDownloadButtons();
    if (isElectron) {
      // 检查更新
      this.checkForUpdates();
      // 监听更新事件
      window.electronAPI.onUpdateAvailable((_, info) => {
        this.updateAvailable = true;
        this.updateInfo = info;
        showNotification(this.t('updateAvailable'), 'info');
      });
      window.electronAPI.onUpdateNotAvailable(() => {
        this.updateAvailable = false;
        this.updateInfo = null;
      });
      window.electronAPI.onUpdateError((_, err) => {
        showNotification(err, 'error');
      });
      window.electronAPI.onDownloadProgress((_, progress) => {
        this.downloadProgress = progress.percent;
        this.updateIcon = 'fa-solid fa-spinner fa-spin';
      });
      window.electronAPI.onUpdateDownloaded(() => {
        this.updateDownloaded = true;
        this.updateIcon = 'fa-solid fa-rocket';
      });
    }
    this.$nextTick(() => {
      this.initPreviewButtons();
    });
    this.$nextTick(() => {               // 保证 DOM 已渲染
      document.addEventListener('click', this._toggleHighlight, false);
    });
    document.documentElement.setAttribute('data-theme', this.systemSettings.theme);
    if (isElectron) {
      window.stopQQBotHandler = this.requestStopQQBotIfRunning;
      window.electronAPI.onWindowState((_, state) => {
        this.isMaximized = state === 'maximized'
      });
    }
    this.initTTSWebSocket();
    this.$nextTick(() => {
      this.generateQRCode(); // 生成二维码
    });
    // 1. 时间触发器（每秒扫一次）
    this.behaviorTimeTimer = setInterval(() => {
      if (!this.behaviorSettings.enabled) return
      const now = new Date()
      const hm = now.toLocaleTimeString('zh-CN', { hour12: false }) // HH:mm:ss
      const d  = now.getDay() // 0=周日
      this.behaviorSettings.behaviorList.forEach(b => {
        if (!b.enabled || b.trigger.type !== 'time') return
        const tv = b.trigger.time.timeValue
        const ds = b.trigger.time.days
        if (tv === hm) {
          if (ds.length === 0 || ds.includes(d)) {
            this.runBehavior(b)
            this.disableOnceBehavior(b)
          }
        }
      })
    }, 1000)

    // 2. 无-input 触发器（每 1s 检查一次）
    this.noInputSec = 0 // 连续无输入秒数
    this.behaviorNoInputTimer = setInterval(() => {
      if (!this.behaviorSettings.enabled) return
      this.behaviorSettings.behaviorList.forEach(b => {
        if (!b.enabled || b.trigger.type !== 'noInput') return
        const need = b.trigger.noInput.latency
        if (this.noInputFlag) {
          this.noInputSec++
          if (this.noInputSec >= need) {
            this.runBehavior(b)
            this.noInputSec = 0 // 触发后重置
          }
        } else {
          this.noInputSec = 0
        }
      })
    }, 1000)

   // 在 mounted() 函数中添加以下代码
  this.cycleTimers = []; // 存储所有周期触发器的定时器

  // 3. 周期触发器（每1秒检查一次）
  this.behaviorCycleTimer = setInterval(() => {
    if (!this.behaviorSettings.enabled) return;
    
    const now = new Date();
    this.behaviorSettings.behaviorList.forEach((b, index) => {
      if (!b.enabled || b.trigger.type !== 'cycle') return;
      
      // 检查是否已有定时器
      if (!this.cycleTimers[index]) {
        this.initCycleTimer(b, index);
      }
    });
  }, 1000); 
  if (this.activeMenu === 'home') this.startDriverGuide();
    this.scanExtensions(); // 扫描扩展
  },
  beforeUnmount() {
    if (isElectron) {
      delete window.stopQQBotHandler;
    }
    if (this.ttsWebSocket) {
      this.ttsWebSocket.close();
    }
    document.removeEventListener('click', this._toggleHighlight, false);
    window.removeEventListener('resize', this.handleResize);
  },
  watch: {
    'readConfig.longText': {
      immediate: true,
      async handler(val) {          // ← 加 async
        await this.$nextTick();     // ← 保证组件完成上一轮渲染
        if (!val?.trim()) {
          this.clearSegments();
          return;
        }
        this.reSegment();
      }
    },
    selectedCodeLang() {
      this.highlightCode();
    },
    modelProviders: {
      deep: true,
      handler(newProviders) {
        const existingIds = new Set(newProviders.map(p => p.id));
        // 自动清理无效的 selectedProvider
        [this.settings, this.reasonerSettings,this.visionSettings,this.KBSettings,this.text2imgSettings,this.ccSettings,this.qcSettings].forEach(config => {
          if (config.selectedProvider && !existingIds.has(config.selectedProvider)) {
            config.selectedProvider = null;
            // 可选项：同时重置相关字段
            config.model = '';
            config.base_url = '';
            config.api_key = '';
          }
          if (!config.selectedProvider && newProviders.length > 0) {
            config.selectedProvider = newProviders[0].id;
          }
        });
        [this.settings, this.reasonerSettings,this.visionSettings,this.KBSettings,this.text2imgSettings,this.ccSettings].forEach(config => {
          if (config.selectedProvider) this.syncProviderConfig(config);
        });
      }
    },
    'systemSettings.theme': {
      handler(newVal) {
        document.documentElement.setAttribute('data-theme', newVal);
        
        // 更新 mermaid 主题
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'loose',
          theme: newVal === 'dark' || newVal === 'midnight' || newVal === 'neon' ? 'dark' : 'default'
        });

        // 完整的主题色映射
        const themeColors = {
          light: '#409eff',      // 默认蓝色
          dark: '#1668dc',       // 深蓝色
          midnight: '#0ea5e9',   // 午夜蓝
          desert: '#d98236',     // 沙漠黄
          neon: '#ff2d95' ,       // 霓虹粉
          marshmallow: '#f5a5c3',  // Marshmallow 粉色
          ink: '#2c3e50',        // 墨水蓝
          party: '#ed7d00',        // 派对黄
        };

        // 获取当前主题色
        const themeColor = themeColors[newVal] || themeColors.light;
        const root = document.documentElement;

        // 设置主色及其衍生色（Element Plus 需要完整的色系）
        root.style.setProperty('--el-color-primary', themeColor);
        root.style.setProperty('--el-color-primary-light-9', this.colorBlend(themeColor, '#ffffff', 0.1));
        root.style.setProperty('--el-color-primary-light-8', this.colorBlend(themeColor, '#ffffff', 0.2));
        root.style.setProperty('--el-color-primary-light-7', this.colorBlend(themeColor, '#ffffff', 0.3));
        root.style.setProperty('--el-color-primary-light-6', this.colorBlend(themeColor, '#ffffff', 0.4));
        root.style.setProperty('--el-color-primary-light-5', this.colorBlend(themeColor, '#ffffff', 0.5));
        root.style.setProperty('--el-color-primary-light-4', this.colorBlend(themeColor, '#ffffff', 0.6));
        root.style.setProperty('--el-color-primary-light-3', this.colorBlend(themeColor, '#ffffff', 0.7));
        root.style.setProperty('--el-color-primary-light-2', this.colorBlend(themeColor, '#ffffff', 0.8));
        root.style.setProperty('--el-color-primary-light-1', this.colorBlend(themeColor, '#ffffff', 0.9));
        root.style.setProperty('--el-color-primary-dark-1', this.colorBlend(themeColor, '#000000', 0.3));
        root.style.setProperty('--el-color-primary-dark-2', this.colorBlend(themeColor, '#000000', 0.2));
        root.style.setProperty('--el-color-primary-dark-3', this.colorBlend(themeColor, '#000000', 0.1));

        // 强制刷新 Element Plus 主题
        if (window.__ELEMENT_PLUS_INSTANCE__) {
          window.__ELEMENT_PLUS_INSTANCE__.config.globalProperties.$ELEMENT.reload();
        }
      },
      immediate: true
    },
    'systemSettings.language': {
      handler(newVal) {
        if (this.isElectron) {
          window.electronAPI.sendLanguage(newVal);
        }
      },
      immediate: true
    },
  },
  computed: {
    sidePanelText() {
      if (this.messages.length === 0) {
        return '';
      }
      
      // 过滤出所有助手消息并按时间倒序排列
      const assistantMessages = this.messages
        .filter(msg => msg.role === 'assistant')
        .reverse();
      
      // 找到第一个非空消息
      for (const msg of assistantMessages) {
        if (msg.pure_content && msg.pure_content.trim() !== '') {
          return msg.pure_content;
        }
      }
      
      // 如果没有找到符合条件的消息
      return '';
    },
  currentViewName() {
    return this.currentExtension ? this.currentExtension.name : this.t('defaultView');
  },
    /* 计算属性：默认模板 */
    defaultSidePanelHTML() {
      // 如果用户已给出自定义模板，就直接用
      if (this.sidePanelHTML) return this.sidePanelHTML;

      return `
        <div class="side-panel-default">
          <div class="side-panel-content markdown-body" v-data-mjx-disabled="true">
            ${this.formatMessage(this.sidePanelText)}
          </div>
        </div>`;
    },
    noInputFlag() {
      return !this.TTSrunning &&
             !this.ASRrunning &&
             !this.isInputting &&
             !this.isTyping
    },
    // 计算处理百分比
    processingPercentage() {
      if (this.totalChunksCount === 0) return 0;
      return Math.round((this.audioChunksCount / this.totalChunksCount) * 100);
    },
    
    // 生成进度文本
    processingProgressText() {
      if (this.totalChunksCount === 0) return this.t('waiting');
      
      return `${this.audioChunksCount} / ${this.totalChunksCount} (${this.processingPercentage}%)`;
    },
    
    // 根据状态设置进度条颜色
    progressStatus() {
      if (this.isReadRunning || this.isConvertingAudio) {
        if (this.processingPercentage >= 90) return 'success';
        if (this.processingPercentage >= 50) return '';
        return 'exception';
      }
      return 'success';
    },

    allChecked: {
      get() {
        return this.textFiles.length > 0 && this.selectedFiles.length === this.textFiles.length;
      },
      set(val) {
        this.selectedFiles = val ? this.textFiles.map(f => f.unique_filename) : [];
      }
    },
    indeterminate() {
      return (
        this.selectedFiles.length > 0 &&
        this.selectedFiles.length < this.textFiles.length
      );
    },
    // 图片全选状态
    allImagesChecked: {
      get() {
        return this.imageFiles.length > 0 && 
              this.selectedImages.length === this.imageFiles.length
      },
      set(val) {
        this.selectedImages = val 
          ? this.imageFiles.map(i => i.unique_filename) 
          : []
      }
    },
    
    // 视频全选状态
    allVideosChecked: {
      get() {
        return this.videoFiles.length > 0 && 
              this.selectedVideos.length === this.videoFiles.length
      },
      set(val) {
        this.selectedVideos = val 
          ? this.videoFiles.map(v => v.unique_filename) 
          : []
      }
    },
    sidebarStyle() {
      return {
        width: this.isMobile ? 
          (this.sidebarVisible ? '200px' : '0') : 
          (this.isCollapse ? '64px' : '200px')
      }
    },
    filteredSeparators() {
      const current = this.qqBotConfig.separators;
      const defaults = this.defaultSeparators;
      const custom = current
        .filter(s => !defaults.some(d => d.value === s))
        .map(s => ({
          label: `(${this.formatSeparator(s)})`,
          value: s
        }));
      return [...this.defaultSeparators, ...custom];
    },
    filteredClaudeModelProviders() {
      let vendors = ["Anthropic", "Deepseek", "siliconflow", "ZhipuAI", "moonshot", "aliyun", "modelscope","302.AI"];
      // this.modelProviders中，vendor在vendors中的，添加到filteredClaudeModelProviders
      return this.modelProviders.filter((item) => vendors.includes(item.vendor));
    },

    // 计算属性，判断配置是否有效
    isQQBotConfigValid() {
        return this.qqBotConfig.appid && this.qqBotConfig.secret;
    },
    isWXBotConfigValid() {
        return this.WXBotConfig.nickNameList && this.WXBotConfig.nickNameList.length > 0;
    },
    isLiveConfigValid() {
        if (this.liveConfig.bilibili_enabled) {
            if(this.liveConfig.bilibili_type === 'web'){
                return this.liveConfig.bilibili_room_id && this.liveConfig.bilibili_room_id.trim() !== '';
            }
            else if(this.liveConfig.bilibili_type === 'open_live'){
                return this.liveConfig.bilibili_ACCESS_KEY_ID !== '' &&
                this.liveConfig.bilibili_SECRET_ACCESS_KEY !== '' &&
                this.liveConfig.bilibili_APP_ID !== '' &&
                this.liveConfig.bilibili_ROOM_OWNER_AUTH_CODE !== '';
            }
        }
        return false;
    },
    updateButtonText() {
      if (this.updateDownloaded) return this.t('installNow');
      if (this.downloadProgress > 0) return this.t('downloading');
      return this.t('updateAvailable');
    },
    allItems() {
      return [
        ...this.files.map(file => ({ ...file, type: 'file' })),
        ...this.images.map(image => ({ ...image, type: 'image' }))
      ];
    },
    sortedConversations() {
      return [...this.conversations].sort((a, b) => b.timestamp - a.timestamp);
    },
    filteredConversations() {
        const keyword = this.searchKeyword.toLowerCase()
        return [...this.conversations]
            .filter(conv => {
                // 匹配标题或消息内容
                const titleMatch = (conv.title || this.t('untitled')).toLowerCase().includes(keyword)
                const contentMatch = conv.messages?.some(msg => 
                    msg.content.toLowerCase().includes(keyword)
                )
                return titleMatch || contentMatch
            })
            .sort((a, b) => b.timestamp - a.timestamp)
    },
    iconClass() {
      return this.isExpanded ? 'fa-solid fa-compress' : 'fa-solid fa-expand';
    },
    hasEnabledA2AServers() {
      return Object.values(this.a2aServers).some(server => server.enabled);
    },
    hasEnabledLLMTools() {
      return this.llmTools.some(tool => tool.enabled);
    },
    hasEnabledKnowledgeBases() {
      return this.knowledgeBases.some(kb => kb.enabled)
    },
    hasEnabledMCPServers() {
      // 检查this.mcpServers中的sever中是否有disable为false的
      return Object.values(this.mcpServers).some(server => !server.disabled);
    },
    hasEnabledHttpTools() {
      return this.customHttpTools.some(tool => tool.enabled);
    },
    hasEnabledComfyUI() {
      return this.workflows.some(tool => tool.enabled);
    },
    hasEnabledStickerPacks() {
      return this.stickerPacks.some(pack => pack.enabled);
    },
    hasFiles() {
      return this.files.length > 0
    },
    hasImages() {
      return this.images.length > 0
    },
    formValid() {
      return !!this.newLLMTool.name && !!this.newLLMTool.type
    },
    defaultBaseURL() {
      switch(this.newLLMTool.type) {
        case 'openai': 
          return 'https://api.openai.com/v1'
        case 'ollama':
          return this.isdocker ? 
            'http://host.docker.internal:11434' : 
            'http://127.0.0.1:11434'
        default:
          return ''
      }
    },
    defaultApikey() {
      switch(this.newLLMTool.type) {
        case 'ollama':
          return 'ollama'
        default:
          return ''
      }
    },
    validProvider() {
      if (!this.newProviderTemp.vendor) return false
      if (this.newProviderTemp.vendor === 'custom') {
        return this.newProviderTemp.url.startsWith('http')
      }
      return true
    },
    vendorOptions() {
      return this.vendorValues.map(value => ({
        label: this.t(`vendor.${value}`), // 使用统一的翻译键
        value
      }));
    },
    MCPvendorOptions() {
      return this.MCPvendorValues.map(value => ({
        label: this.t(`MCPvendor.${value}`), // 使用统一的翻译键
        value
      }));
    },
    PromptOptions() {
      return this.promptValues.map(value => ({
        label: this.t(`prompt.${value}`), // 使用统一的翻译键
        value
      }));
    },
    CardOptions() {
      return this.cardValues.map(value => ({
        label: this.t(`card.${value}`), // 使用统一的翻译键
        value
      }));
    },
    themeOptions() {
      return this.themeValues.map(value => ({
        label: this.t(`theme.${value}`),
        value // 保持原始值（推荐）
      }));
    },
    hasAgentChanges() {
      return this.mainAgent !== 'super-model' || 
        Object.values(this.agents).some(a => a.enabled)
    },
    // 获取所有唯一的语言
    uniqueLanguages() {
      const languages = [...new Set(this.edgettsvoices.map(voice => voice.language))];
      return languages.sort();
    },
    
    // 根据选择的语言获取可用的性别
    uniqueGenders() {
      const voicesForLanguage = this.edgettsvoices.filter(voice => 
        voice.language === this.edgettsLanguage
      );
      const genders = [...new Set(voicesForLanguage.map(voice => voice.gender))];
      return genders.sort();
    },
    
    // 根据选择的语言和性别过滤语音
    filteredVoices() {
      return this.edgettsvoices.filter(voice => 
        voice.language === this.edgettsLanguage && 
        voice.gender === this.edgettsGender
      );
    },
    uniqueNewLanguages() {
      const languages = [...new Set(this.edgettsvoices.map(voice => voice.language))];
      return languages.sort();
    },
    uniqueNewGenders() {
      const voicesForLanguage = this.edgettsvoices.filter(voice => 
        voice.language === this.newTTSConfig.edgettsLanguage
      );
      const genders = [...new Set(voicesForLanguage.map(voice => voice.gender))];
      return genders.sort();
    },
    filteredNewVoices() {
      return this.edgettsvoices.filter(voice => 
        voice.language === this.newTTSConfig.edgettsLanguage && 
        voice.gender === this.newTTSConfig.edgettsGender
      );
    },
    selectedVendor() {
      return this.modelProviders.find(
        p => p.id === this.settings.selectedProvider
      );
    },
  },
  methods: vue_methods
});

function showNotification(message, type = 'success') {
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  // 强制重绘确保动画生效
  void notification.offsetWidth;
  
  notification.classList.add('show');

  // 设置显示时间：错误提示显示5秒，其他提示显示2秒
  const duration = type === 'error' ? 5000 : 2000;

  setTimeout(() => {
    notification.classList.remove('show');
    notification.classList.add('hide');
    setTimeout(() => notification.remove(), 400);
  }, duration);
}

function removeNonAsciiTags(html) {
  // 匹配所有标签（包括开始标签和结束标签）
  // 例如：<旁白> 和 </旁白>
  const regex = /<\/?([^\s>]+)[^>]*>/g;
  
  return html.replace(regex, (match, tagName) => {
    // 检查标签名是否包含非 ASCII 字符
    const hasNonAscii = [...tagName].some(char => char.charCodeAt(0) > 127);
    
    // 如果标签名包含非 ASCII 字符，删除标签（但保留内容）
    if (hasNonAscii) {
      return '';
    }
    
    // 否则，保留标签
    return match;
  });
}

// 修改图标注册方式（完整示例）
app.use(ElementPlus);

// 正确注册所有图标（一次性循环注册）
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}


// 挂载应用
app.mount('#app');
