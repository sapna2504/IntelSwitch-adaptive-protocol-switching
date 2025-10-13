/**
 * The copyright in this software is being made available under the BSD License,
 * included below. This software may be subject to other third party and contributor
 * rights, including patent rights, and no such rights are granted under this license.
 *
 * Copyright (c) 2013, Dash Industry Forum.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice, this
 *  list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation and/or
 *  other materials provided with the distribution.
 *  * Neither the name of Dash Industry Forum nor the names of its
 *  contributors may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 *  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 *  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 *  NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

import ABRRulesCollection from '../rules/abr/ABRRulesCollection.js';
import Constants from '../constants/Constants.js';
import MetricsConstants from '../constants/MetricsConstants.js';
import FragmentModel from '../models/FragmentModel.js';
import EventBus from '../../core/EventBus.js';
import Events from '../../core/events/Events.js';
import FactoryMaker from '../../core/FactoryMaker.js';
import RulesContext from '../rules/RulesContext.js';
import SwitchRequest from '../rules/SwitchRequest.js';
import SwitchRequestHistory from '../rules/SwitchRequestHistory.js';
import DroppedFramesHistory from '../rules/DroppedFramesHistory.js';
import Debug from '../../core/Debug.js';
import MediaPlayerEvents from '../MediaPlayerEvents.js';
import ThroughputModel from '../models/ThroughputModel.js';
import MetricsModel from '../models/MetricsModel.js';

const DEFAULT_VIDEO_BITRATE = 1000;
const DEFAULT_BITRATE = 100;

function AbrController() {

    const context = this.context;
    const debug = Debug(context).getInstance();
    const eventBus = EventBus(context).getInstance();
    let lastBitrate = -1; //newly added
    let bitrateVariation = -1; // newly added
    let playbackEnded = false; //newly added
    let updatedProtocol = 'http2';
    let lastUpdatedProtocol;
    let protocol = 'http2';
    let lastProtocolDecisionIndex = {}; // mediaType_streamId => index
    
    
    let instance,
        logger,
        abrRulesCollection,
        streamController,
        streamProcessorDict,
        abandonmentStateDict,
        abandonmentTimeout,
        windowResizeEventCalled,
        adapter,
        videoModel,
        mediaPlayerModel,
        customParametersModel,
        cmsdModel,
        domStorage,
        playbackRepresentationId,
        switchRequestHistory,
        droppedFramesHistory,
        throughputController,
        throughputModel,
        dashMetrics,
        bufferLevel,
        metricsModel,
        settings;

    function setup() {
        logger = debug.getLogger(instance); 
        resetInitialSettings();
        
    }

    /**
     * Initialize everything that is not period specific. We only have one instance of the ABR Controller for all periods.
     */
    function initialize() {
        droppedFramesHistory = DroppedFramesHistory(context).create();
        switchRequestHistory = SwitchRequestHistory(context).create();
        abrRulesCollection = ABRRulesCollection(context).create({
            dashMetrics,
            customParametersModel,
            mediaPlayerModel,
            settings
        });
        abrRulesCollection.initialize();

        eventBus.on(MediaPlayerEvents.QUALITY_CHANGE_RENDERED, _onQualityChangeRendered, instance);
        eventBus.on(MediaPlayerEvents.METRIC_ADDED, _onMetricAdded, instance);
        eventBus.on(Events.LOADING_PROGRESS, _onFragmentLoadProgress, instance);
        eventBus.on(MediaPlayerEvents.PLAYBACK_PLAYING, _onPlaybackInitiated, instance);
        eventBus.on(MediaPlayerEvents.PLAYBACK_ENDED, _onPlaybackEnded, instance); 
    	
    }
    

    /**
     * Whenever a StreamProcessor is created it is added to the list of streamProcessorDict
     * In addition, the corresponding objects for this object and its stream id are created
     * @param {object} type
     * @param {object} streamProcessor
     */
    function registerStreamType(type, streamProcessor) {
        const streamId = streamProcessor.getStreamInfo().id;

        if (!streamProcessorDict[streamId]) {
            streamProcessorDict[streamId] = {};
        }
        streamProcessorDict[streamId][type] = streamProcessor;

        if (!abandonmentStateDict[streamId]) {
            abandonmentStateDict[streamId] = {};
        }
        abandonmentStateDict[streamId][type] = {};
        abandonmentStateDict[streamId][type].state = MetricsConstants.ALLOW_LOAD;

        // Do not change current value if it has been set before
        const currentState = abrRulesCollection.getBolaState(type)
        if (currentState === undefined) {
            abrRulesCollection.setBolaState(type, settings.get().streaming.abr.rules.bolaRule.active && !_shouldApplyDynamicAbrStrategy());
        }

    }

    /**
     * Remove all parameters that belong to a specific period
     * @param {string} streamId
     * @param {string} type
     */
    function unRegisterStreamType(streamId, type) {
        try {
            if (streamProcessorDict[streamId] && streamProcessorDict[streamId][type]) {
                delete streamProcessorDict[streamId][type];
            }

            if (abandonmentStateDict[streamId] && abandonmentStateDict[streamId][type]) {
                delete abandonmentStateDict[streamId][type];
            }

        } catch (e) {

        }
    }

    function resetInitialSettings() {
        abandonmentStateDict = {};
        streamProcessorDict = {};

        if (windowResizeEventCalled === undefined) {
            windowResizeEventCalled = false;
        }
        if (droppedFramesHistory) {
            droppedFramesHistory.reset();
        }

        if (switchRequestHistory) {
            switchRequestHistory.reset();
        }

        playbackRepresentationId = undefined;
        droppedFramesHistory = undefined;
        switchRequestHistory = undefined;
        clearTimeout(abandonmentTimeout);
        abandonmentTimeout = null;
    }

    function reset() {

        resetInitialSettings();

        eventBus.off(MediaPlayerEvents.QUALITY_CHANGE_RENDERED, _onQualityChangeRendered, instance);
        eventBus.off(MediaPlayerEvents.METRIC_ADDED, _onMetricAdded, instance);
        eventBus.off(Events.LOADING_PROGRESS, _onFragmentLoadProgress, instance);

        if (abrRulesCollection) {
            abrRulesCollection.reset();
        }
    }

    function setConfig(config) {
        if (!config) {
            return;
        }
	//console.log("the setconfig in abrcontroller is", config);
        if (config.streamController) {
            streamController = config.streamController;
        }
        if (config.throughputController) {
            throughputController = config.throughputController;
        }
        if (config.domStorage) {
            domStorage = config.domStorage;
        }
        if (config.mediaPlayerModel) {
            mediaPlayerModel = config.mediaPlayerModel;
        }
        if (config.customParametersModel) {
            customParametersModel = config.customParametersModel;
        }
        if (config.cmsdModel) {
            cmsdModel = config.cmsdModel
        }
        if (config.dashMetrics) {
            dashMetrics = config.dashMetrics;
        }
        if (config.adapter) {
            adapter = config.adapter;
        }
        if (config.videoModel) {
            videoModel = config.videoModel;
        }
        if (config.MetricsModel) {
            MetricsModel = config.MetricsModel;
        }
        if (config.settings) {
            settings = config.settings;
        }
    }

    function getOptimalRepresentationForBitrate(mediaInfo, bitrate, includeCompatibleMediaInfos = true) {
        const possibleVoRepresentations = getPossibleVoRepresentationsFilteredBySettings(mediaInfo, includeCompatibleMediaInfos);

        if (!possibleVoRepresentations || possibleVoRepresentations.length === 0) {
            return null;
        }

        // If bitrate should be as small as possible return the Representation with the lowest bitrate
        const smallestRepresentation = possibleVoRepresentations.reduce((a, b) => {
            return a.bandwidth < b.bandwidth ? a : b;
        })
        if (bitrate <= 0) {
            return smallestRepresentation
        }

        // Get all Representations that have lower or equal bitrate than our target bitrate
        const targetRepresentations = possibleVoRepresentations.filter((rep) => {
            return rep.bitrateInKbit <= bitrate
        });

        if (!targetRepresentations || targetRepresentations.length === 0) {
            return smallestRepresentation
        }

        return targetRepresentations.reduce((max, curr) => {
            return (curr.absoluteIndex > max.absoluteIndex) ? curr : max;
        })

    }

    function getRepresentationByAbsoluteIndex(absoluteIndex, mediaInfo, includeCompatibleMediaInfos = true) {
        if (isNaN(absoluteIndex) || absoluteIndex < 0) {
            return null;
        }

        const possibleVoRepresentations = getPossibleVoRepresentationsFilteredBySettings(mediaInfo, includeCompatibleMediaInfos);

        return possibleVoRepresentations.find((rep) => {
            return rep.absoluteIndex === absoluteIndex
        })
    }

    function getPossibleVoRepresentations(mediaInfo, includeCompatibleMediaInfos = true) {
        return _getPossibleVoRepresentations(mediaInfo, includeCompatibleMediaInfos)
    }

    function getPossibleVoRepresentationsFilteredBySettings(mediaInfo, includeCompatibleMediaInfos = true) {
        let voRepresentations = _getPossibleVoRepresentations(mediaInfo, includeCompatibleMediaInfos);

        // Filter the list of options based on the provided settings
        voRepresentations = _filterByAllowedSettings(voRepresentations)

        return voRepresentations;
    }

    function _getPossibleVoRepresentations(mediaInfo, includeCompatibleMediaInfos) {
        let voRepresentations = [];
        if (!mediaInfo) {
            return voRepresentations;
        }

        const mediaInfos = _getPossibleMediaInfos(mediaInfo)
        mediaInfos.forEach((mediaInfo) => {
            let currentVoRepresentations = adapter.getVoRepresentations(mediaInfo);

            if (currentVoRepresentations && currentVoRepresentations.length > 0) {
                voRepresentations = voRepresentations.concat(currentVoRepresentations)
            }
        })

        // Now sort by quality (usually simply by bitrate)
        voRepresentations = _sortRepresentationsByQuality(voRepresentations);

        // Add an absolute index
        voRepresentations.forEach((rep, index) => {
            rep.absoluteIndex = index
        })

        // Filter the Representations in case we do not want to include compatible Media Infos
        // We can not apply the filter before otherwise the absolute index would be wrong
        if (!includeCompatibleMediaInfos) {
            voRepresentations = voRepresentations.filter((rep) => {
                return adapter.areMediaInfosEqual(rep.mediaInfo, mediaInfo);
            })
        }

        return voRepresentations
    }

    function _getPossibleMediaInfos(mediaInfo) {
        try {
            const possibleMediaInfos = [];

            if (mediaInfo) {
                possibleMediaInfos.push(mediaInfo);
            }

            // If AS switching is disabled return only the current MediaInfo
            if (!settings.get().streaming.abr.enableSupplementalPropertyAdaptationSetSwitching
                || !mediaInfo.adaptationSetSwitchingCompatibleIds
                || mediaInfo.adaptationSetSwitchingCompatibleIds.length === 0) {
                return possibleMediaInfos
            }

            // Otherwise add everything that is compatible
            const mediaInfoArr = streamProcessorDict[mediaInfo.streamInfo.id][mediaInfo.type].getAllMediaInfos()
            const compatibleMediaInfos = mediaInfoArr.filter((entry) => {
                return mediaInfo.adaptationSetSwitchingCompatibleIds.includes(entry.id)
            })

            return possibleMediaInfos.concat(compatibleMediaInfos);
        } catch (e) {
            return [mediaInfo]
        }
    }

    /**
     * @param {Representation[]} voRepresentations
     * @return {Representation[]}
     */
    function _filterByAllowedSettings(voRepresentations) {
        try {
            voRepresentations = _filterByPossibleBitrate(voRepresentations);
            voRepresentations = _filterByPortalSize(voRepresentations);
            voRepresentations = _filterByCmsdMaxBitrate(voRepresentations);

            return voRepresentations;
        } catch (e) {
            logger.error(e);
            return voRepresentations
        }
    }

    /**
     * Returns all RepresentationInfo objects that have at least one bitrate that fulfills the constraint
     * @param {Representation[]} voRepresentations
     * @return {Representation[]}
     */
    function _filterByPossibleBitrate(voRepresentations) {
        try {
            const filteredArray = voRepresentations.filter((voRepresentation) => {
                const type = voRepresentation.mediaInfo.type;
                const representationBitrate = voRepresentation.bitrateInKbit;
                const maxBitrate = mediaPlayerModel.getAbrBitrateParameter('maxBitrate', type);
                const minBitrate = mediaPlayerModel.getAbrBitrateParameter('minBitrate', type);

                if (maxBitrate > -1 && representationBitrate > maxBitrate) {
                    return false;
                }

                return !(minBitrate > -1 && representationBitrate < minBitrate);
            })

            if (filteredArray.length > 0) {
                return filteredArray
            }

            return voRepresentations
        } catch (e) {
            logger.error(e);
            return voRepresentations
        }
    }

    /**
     * @param {Representation[]} voRepresentations
     * @return {Representation[]}
     * @private
     */
    function _filterByPortalSize(voRepresentations) {
        try {
            if (!settings.get().streaming.abr.limitBitrateByPortal) {
                return voRepresentations;
            }

            const { elementWidth } = videoModel.getVideoElementSize();

            const filteredArray = voRepresentations.filter((voRepresentation) => {
                return voRepresentation.mediaInfo.type !== Constants.VIDEO || voRepresentation.width <= elementWidth;
            })

            if (filteredArray.length > 0) {
                return filteredArray
            }

            return voRepresentations
        } catch (e) {
            logger.error(e);
            return voRepresentations
        }
    }

    /**
     * @param {Representation[]} voRepresentations
     * @return {Representation[]}
     */
    function _filterByCmsdMaxBitrate(voRepresentations) {
        try {
            // Check CMSD max suggested bitrate only for video segments
            if (!settings.get().streaming.cmsd.enabled || !settings.get().streaming.cmsd.abr.applyMb) {
                return voRepresentations
            }

            const filteredArray = voRepresentations.filter((voRepresentation) => {
                const type = voRepresentation.mediaInfo.type;
                let maxCmsdBitrate = cmsdModel.getMaxBitrate(type);

                if (type !== Constants.VIDEO || maxCmsdBitrate < 0) {
                    return true
                }
                // Subtract audio bitrate
                const streamId = voRepresentation.mediaInfo.streamInfo.id;
                const streamProcessor = streamProcessorDict[streamId][Constants.AUDIO];
                const representation = streamProcessor.getRepresentation();
                const audioBitrate = representation.bitrateInKbit;
                maxCmsdBitrate -= audioBitrate ? audioBitrate : 0;
                return voRepresentation.bitrateInKbit <= maxCmsdBitrate
            })

            if (filteredArray.length > 0) {
                return filteredArray
            }

            return voRepresentations
        } catch (e) {
            logger.error(e);
            return voRepresentations
        }
    }

    function _sortRepresentationsByQuality(voRepresentations) {
        if (_shouldSortByQualityRankingAttribute(voRepresentations)) {
            voRepresentations = _sortByQualityRankingAttribute(voRepresentations)
        } else {
            voRepresentations = _sortByDefaultParameters(voRepresentations)
        }

        return voRepresentations
    }

    function _shouldSortByQualityRankingAttribute(voRepresentations) {
        let firstMediaInfo = null;
        const filteredRepresentations = voRepresentations.filter((rep) => {
            if (!firstMediaInfo) {
                firstMediaInfo = rep.mediaInfo;
            }
            return !isNaN(rep.qualityRanking) && adapter.areMediaInfosEqual(firstMediaInfo, rep.mediaInfo);
        })

        return filteredRepresentations.length === voRepresentations.length
    }

    function _sortByQualityRankingAttribute(voRepresentations) {
        voRepresentations.sort((a, b) => {
            return b.qualityRanking - a.qualityRanking;
        })

        return voRepresentations
    }


    function _sortByDefaultParameters(voRepresentations) {
        voRepresentations.sort((a, b) => {

            // In case both Representations are coming from the same MediaInfo then choose the one with the highest resolution and highest bitrate
            if (adapter.areMediaInfosEqual(a.mediaInfo, b.mediaInfo)) {
                if (!isNaN(a.pixelsPerSecond) && !isNaN(b.pixelsPerSecond) && a.pixelsPerSecond !== b.pixelsPerSecond) {
                    return a.pixelsPerSecond - b.pixelsPerSecond
                } else {
                    return a.bandwidth - b.bandwidth
                }
            }

            // In case the Representations are coming from different MediaInfos they might have different codecs. The bandwidth is not a good indicator, use bits per pixel instead
            else {
                if (!isNaN(a.pixelsPerSecond) && !isNaN(b.pixelsPerSecond) && a.pixelsPerSecond !== b.pixelsPerSecond) {
                    return a.pixelsPerSecond - b.pixelsPerSecond
                } else if (!isNaN(a.bitsPerPixel) && !isNaN(b.bitsPerPixel)) {
                    return b.bitsPerPixel - a.bitsPerPixel
                } else {
                    return a.bandwidth - b.bandwidth
                }
            }
        })

        return voRepresentations
    }
    
    function getUpdatedProtocol(){
        if (updatedProtocol !== undefined) {
            lastUpdatedProtocol = updatedProtocol;
        }
        //console.log("theeeeeeeeeeeeeeeeeeee protooooooooooooooooo :", updatedProtocol);
        return updatedProtocol !== undefined ? updatedProtocol : lastUpdatedProtocol;
    }
    
    async function setUpdatedProtocol(protocol){
        updatedProtocol = protocol;  //newly added
    }

    /**
     * While fragment loading is in progress we check if we might need to abort the request
     * @param {object} e
     * @private
     */
    async function _onFragmentLoadProgress(e) {
        const type = e.request.mediaType;
        const streamId = e.streamId;
        const segmentIndex = e.request.index;

        const key = `${type}_${streamId}`;
        if (lastProtocolDecisionIndex[key] === segmentIndex) {
            // Already queried RL for this segment
            return;
        }
        lastProtocolDecisionIndex[key] = segmentIndex;

        
        const streamProcessor = streamProcessorDict[streamId][type];
        const fragmentModel = streamProcessor.getFragmentModel();
        const requestList = fragmentModel.getRequests({
    	    state: FragmentModel.FRAGMENT_MODEL_LOADING,
            index: e.request.index
        });
        if (requestList && requestList.length > 0 && type === 'video') {
           const originalRequest = requestList[0];


            // Assign protocol dynamically based on some logic
     
            try {
                updatedProtocol = await getNextSegmentProtocol(type, streamId, segmentIndex);
                originalRequest.protocol = updatedProtocol;
                setUpdatedProtocol(updatedProtocol);
                getUpdatedProtocol();
            } catch (err) {
                console.error("Error getting protocol decision from RL:", err);
                updatedProtocol = 'http2'; // fallback default
            }

        lastUpdatedProtocol = updatedProtocol;
        
        //const updated_protocol = getNextSegmentProtocol(type, streamId); //newly added
        //request.protocol = updated_protocol;
        //console.log("the e.request.protocol is ", originalRequest.protocol);

	
	
        if (!type || !streamId || !streamProcessorDict[streamId] || !settings.get().streaming.abr.autoSwitchBitrate[type]) {
            return;
        }

        
        if (!streamProcessor) {
            return;
        }

        const rulesContext = RulesContext(context).create({
            abrController: instance,
            streamProcessor,
            currentRequest: e.request,
            throughputController,
            adapter,
            videoModel
        });
        const switchRequest = abrRulesCollection.shouldAbandonFragment(rulesContext);
       
      
        if (switchRequest && switchRequest.representation !== SwitchRequest.NO_CHANGE) {
            _onSegmentDownloadShouldBeAbandoned(e, streamId, type, streamProcessor, switchRequest);
        }	    	     
    }
    }

    function _onSegmentDownloadShouldBeAbandoned(e, streamId, type, streamProcessor, switchRequest) {
        const fragmentModel = streamProcessor.getFragmentModel();
        const request = fragmentModel.getRequests({
            state: FragmentModel.FRAGMENT_MODEL_LOADING,
            index: e.request.index
        })[0];
        if (request) {
            abandonmentStateDict[streamId][type].state = MetricsConstants.ABANDON_LOAD;
            switchRequestHistory.reset();
            setPlaybackQuality(type, streamController.getActiveStreamInfo(), switchRequest.representation, switchRequest.reason);

            clearTimeout(abandonmentTimeout);
            abandonmentTimeout = setTimeout(
                () => {
                    abandonmentStateDict[streamId][type].state = MetricsConstants.ALLOW_LOAD;
                    abandonmentTimeout = null;
                },
                settings.get().streaming.abandonLoadTimeout
            );
        }
    }

    /**
     * Update dropped frames history when the quality was changed
     * @param {object} e
     * @private
     */
    function _onQualityChangeRendered(e) {
        if (e.mediaType === Constants.VIDEO) {
            if (playbackRepresentationId !== undefined) {
                droppedFramesHistory.push(e.streamId, playbackRepresentationId, videoModel.getPlaybackQuality());
            }
            playbackRepresentationId = e.newRepresentation.id;
        }
    }

    /**
     * When the buffer level is updated we check if we need to change the ABR strategy
     * @param e
     * @private
     */
    function _onMetricAdded(e) {
        if (_shouldApplyDynamicAbrStrategy()
            && e.metric === MetricsConstants.BUFFER_LEVEL
            && (e.mediaType === Constants.AUDIO || e.mediaType === Constants.VIDEO)) {
            _updateDynamicAbrStrategy(e.mediaType, 0.001 * e.value.level);
        }
    }
    /*----------------newly added--------------------*/
    function _onPlaybackEnded(e) {
    	playbackEnded = true;
    } 
    
    function _onPlaybackInitiated(e) {
    	playbackEnded = false;
    } 
    
    function hasPlaybackEnded() {
        return playbackEnded;
    }
    /*----------------newly added--------------------*/

    /**
     * Returns the initial bitrate for a specific media type
     * @param {string} type
     * @returns {number} A value of the initial bitrate, kbps
     * @memberof AbrController#
     */
    function getInitialBitrateFor(type) {

        if (type === Constants.TEXT) {
            return NaN;
        }

        let configBitrate = mediaPlayerModel.getAbrBitrateParameter('initialBitrate', type);
        if (configBitrate > 0) {
            return configBitrate;
        }

        let savedBitrate = NaN;
        if (domStorage && domStorage.hasOwnProperty('getSavedBitrateSettings')) {
            savedBitrate = domStorage.getSavedBitrateSettings(type);
        }
        if (!isNaN(savedBitrate)) {
            return savedBitrate
        }

        const averageThroughput = throughputController.getAverageThroughput(type);
        if (!isNaN(averageThroughput) && averageThroughput > 0) {
            return averageThroughput
        }

        return (type === Constants.VIDEO) ? DEFAULT_VIDEO_BITRATE : DEFAULT_BITRATE;
    }

    /**
     * This function is called by the scheduleControllers to check if the quality should be changed.
     * Consider this the main entry point for the ABR decision logic
     * @param {string} type
     * @param {string} streamId
     */
    function checkPlaybackQuality(type, streamId) {
        try {
            if (!type || !streamProcessorDict || !streamProcessorDict[streamId] || !streamProcessorDict[streamId][type]) {
                return false;
            }

            if (droppedFramesHistory) {
                const playbackQuality = videoModel.getPlaybackQuality();
                if (playbackQuality) {
                    droppedFramesHistory.push(streamId, playbackRepresentationId, playbackQuality);
                }
            }

            if (!settings.get().streaming.abr.autoSwitchBitrate[type]) {
                return false;
            }

            const streamProcessor = streamProcessorDict[streamId][type];
            const currentRepresentation = streamProcessor.getRepresentation();
            const rulesContext = RulesContext(context).create({
                abrController: instance,
                throughputController,
                switchRequestHistory,
                droppedFramesHistory,
                streamProcessor,
                adapter,
                videoModel
            });
            const switchRequest = abrRulesCollection.getBestPossibleSwitchRequest(rulesContext);

            if (!switchRequest || !switchRequest.representation) {
                return false;
            }

            let newRepresentation = switchRequest.representation;
            switchRequestHistory.push({
                currentRepresentation,
                newRepresentation
            });

            if (newRepresentation.id !== currentRepresentation.id && (abandonmentStateDict[streamId][type].state === MetricsConstants.ALLOW_LOAD || newRepresentation.absoluteIndex < currentRepresentation.absoluteIndex)) {
                _changeQuality(currentRepresentation, newRepresentation, switchRequest.reason);
                return true;
            }
	    
            return false;
        } catch (e) {
            logger.error(e);
            return false;
        }
    }

    /**
     * Sets the new playback quality. Starts from index 0.
     * If the index of the new quality is the same as the old one changeQuality will not be called.
     * @param {string} type
     * @param {object} streamInfo
     * @param {Representation} representation
     * @param {string} reason
     * @param {string} rule
     */
    function setPlaybackQuality(type, streamInfo, representation, reason = {}) {
        if (!streamInfo || !streamInfo.id || !type || !streamProcessorDict || !streamProcessorDict[streamInfo.id] || !streamProcessorDict[streamInfo.id][type] || !representation) {
            return;
        }

        const streamProcessor = streamProcessorDict[streamInfo.id][type];
        const currentRepresentation = streamProcessor.getRepresentation();

        if (!currentRepresentation || representation.id !== currentRepresentation.id) {
            _changeQuality(currentRepresentation, representation, reason);
        }
    }

    /**
     *
     * @param {string} streamId
     * @param {type} type
     * @return {*|null}
     */
    function getAbandonmentStateFor(streamId, type) {
        return abandonmentStateDict[streamId] && abandonmentStateDict[streamId][type] ? abandonmentStateDict[streamId][type].state : null;
    }


    /**
     * Changes the internal qualityDict values according to the new quality
     * @param {Representation} oldRepresentation
     * @param {Representation} newRepresentation
     * @param {string} reason
     * @private
     */
    function _changeQuality(oldRepresentation, newRepresentation, reason) {
        const streamId = newRepresentation.mediaInfo.streamInfo.id;
        const type = newRepresentation.mediaInfo.type;
        //console.log("the dashmetrics in abr controller is: ", dashMetrics);
        //console.log("the current request is using dash metrics: ", dashMetrics.getCurrentDroppedFrames(type)); 
        if (type && streamProcessorDict[streamId] && streamProcessorDict[streamId][type]) {
            const streamInfo = streamProcessorDict[streamId][type].getStreamInfo();
            const bufferLevel = dashMetrics.getCurrentBufferLevel(type);
            const isAdaptationSetSwitch = oldRepresentation !== null && !adapter.areMediaInfosEqual(oldRepresentation.mediaInfo, newRepresentation.mediaInfo);

            const oldBitrate = oldRepresentation ? oldRepresentation.bitrateInKbit : 0;
            logger.info(`[AbrController]: Switching quality in period ${streamId} for media type ${type}. Switch from bitrate ${oldBitrate} to bitrate ${newRepresentation.bitrateInKbit}. Current buffer level: ${bufferLevel}. Reason:` + (reason ? JSON.stringify(reason) : '/'));

            eventBus.trigger(Events.QUALITY_CHANGE_REQUESTED,
                {
                    oldRepresentation: oldRepresentation,
                    newRepresentation: newRepresentation,
                    reason,
                    streamInfo,
                    mediaType: type,
                    isAdaptationSetSwitch
                },
                { streamId: streamInfo.id, mediaType: type }
            );
            const bitrate = throughputController.getAverageThroughput(type);
            if (!isNaN(bitrate)) {
                domStorage.setSavedBitrateSettings(type, bitrate);
            }
        }
    }


    /**
     * If both BOLA and Throughput Rule are active we switch dynamically between both of them
     * @returns {boolean}
     * @private
     */
    function _shouldApplyDynamicAbrStrategy() {
        return settings.get().streaming.abr.rules.bolaRule.active && settings.get().streaming.abr.rules.throughputRule.active
    }

    /**
     * Switch between BOLA and ThroughputRule
     * @param mediaType
     * @param bufferLevel
     * @private
     */
    function _updateDynamicAbrStrategy(mediaType, bufferLevel) {
        try {
            const bufferTimeDefault = mediaPlayerModel.getBufferTimeDefault();
            const switchOnThreshold = bufferTimeDefault;
            const switchOffThreshold = 0.5 * bufferTimeDefault;

            const isUsingBolaRule = abrRulesCollection.getBolaState(mediaType)
            const shouldUseBolaRule = bufferLevel >= (isUsingBolaRule ? switchOffThreshold : switchOnThreshold); // use hysteresis to avoid oscillating rules
            abrRulesCollection.setBolaState(mediaType, shouldUseBolaRule);

            if (shouldUseBolaRule !== isUsingBolaRule) {
                if (shouldUseBolaRule) {
                    logger.info('[' + mediaType + '] switching from throughput to buffer occupancy ABR rule (buffer: ' + bufferLevel.toFixed(3) + ').');
                } else {
                    logger.info('[' + mediaType + '] switching from buffer occupancy to throughput ABR rule (buffer: ' + bufferLevel.toFixed(3) + ').');
                }
            }
        } catch (e) {
            logger.error(e);
        }
    }

    /**
     * Checks if the provided Representation has the lowest possible quality
     * @param representation
     * @returns {boolean}
     */
    function isPlayingAtLowestQuality(representation) {
        const voRepresentations = getPossibleVoRepresentationsFilteredBySettings(representation.mediaInfo, true);

        return voRepresentations[0].id === representation.id
    }

    /**
     * Checks if the provided Representation has the highest possible quality
     * @param representation
     * @returns {boolean}
     */
    function isPlayingAtTopQuality(representation) {
        if (!representation) {
            return true;
        }
        const voRepresentations = getPossibleVoRepresentationsFilteredBySettings(representation.mediaInfo, true);

        return voRepresentations[voRepresentations.length - 1].id === representation.id;
    }

    function setWindowResizeEventCalled(value) {
        windowResizeEventCalled = value;
    }

    function clearDataForStream(streamId) {
        if (droppedFramesHistory) {
            droppedFramesHistory.clearForStream(streamId);
        }
        if (streamProcessorDict[streamId]) {
            delete streamProcessorDict[streamId];
        }
        if (switchRequestHistory) {
            switchRequestHistory.clearForStream(streamId);
        }
        if (abandonmentStateDict[streamId]) {
            delete abandonmentStateDict[streamId];
        }
    }


    async function queryRLServerAsync(data) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            xhr.open("POST", "http://192.168.3.173:8111", true);

            xhr.setRequestHeader('Content-Type', 'application/json');

            xhr.onreadystatechange = () => {
                if (xhr.readyState === 4) {
                    if (xhr.status === 200) {
                        try {
                            const rlResponse = JSON.parse(xhr.responseText);
                            resolve(rlResponse);
                        } catch (e) {
                            reject(e);
                        }
                    } else {
                        reject(new Error("HTTP status " + xhr.status));
                    }
                }
            };

            xhr.onerror = () => reject(new Error("Network error"));
            //console.log("Sending RL data:", JSON.stringify(data));
            xhr.send(JSON.stringify(data));
        });
    }


    async function getNextSegmentProtocol(type, streamId, segmentIndex) {
        // Gather data to send
        //4g-5g [45,45,44,35,34,24,37,19,16,22,23,9,0,0,0,6,14,22,15,23,10,1,4,57,18,20,26,34,38,28,5,27,54,51,43,37,32,30,5,34,36,30,30,47,45,41,50,45,49,23,45,38,64,74,85,117,108,95,114,107,130,81,79,52,44,11,16,6,67,22,37,23,45,38,36,34,31,31,31,31,32,24,37,31,18,29,42,34,31,12,31,33,28.5,19,12.5,19.5,26,28.5,47.5,58.5,53.5,52.5,46.5,34.5,30,33,37,26,18,34,39.5,28.5,31,34.5,29.5,30,30.5,31,33,34,34.5,32,33.5,36.5,30,30,35.5,33,28.5,19.5,23.5,39,41,38.5,34,26,25,30,25.5,17.5,21,26,27,26.5,25.5,26,32,45.5, 53.5, 55]
        //airtel
        //const networkthroughput= [0.492416, 56.905184, 56.905184, 56.905184, 56.905184, 67.444224, 59.656928, 46.381088, 14.994944, 31.224896, 39.051232, 37.85296, 16.082528, 10.49952, 23.280896, 17.156, 12.950912, 23.035328, 24.527744, 18.55872, 10.657952, 19.184384, 23.383392, 33.235264, 29.576416, 38.172992, 35.814144, 29.179008, 27.025728, 24.171328, 14.574208, 28.0416, 15.534176, 32.474112, 17.761568, 23.700384, 4.6696, 1.410624, 0.370912, 2.647424, 0.924608, 0.874432, 1.359584, 0.438816, 2.495904, 5.069984, 3.581312, 1.98256, 1.609408, 5.648576, 10.82704, 9.14672, 8.843616, 9.351968, 5.409952, 9.444832, 7.272192, 8.581376, 8.444512, 9.347776, 8.298688, 6.209536, 2.830208, 2.03584, 3.230464, 3.643328, 2.928736, 3.065408, 3.43504, 6.148864, 4.709536, 3.973472, 3.751072, 2.49056, 3.43344, 11.612352, 10.85248, 12.356864, 14.487968, 27.623072, 26.863936, 25.068416, 20.367296, 18.239168, 22.943616, 19.098144, 8.961632, 22.169664, 11.808928, 5.321216, 6.131904, 7.50816, 7.528448, 6.372128, 5.369696, 5.550336, 6.29056, 7.341056, 7.384832, 7.658272, 6.640128, 7.853664, 9.383872, 9.389632, 9.15068811,73.8016, 11.97344,11.446432,10.239744,12.740256,14.891968,13.789248,16.43008,12.18816,11.649504,15.916192,14.858112,17.164224,16.333568,20.462848,17.225728,19.701056,23.231584,23.085312,20.966592,25.1216,28.002336,28.549408,49.681408,49.263936,62.114848,52.197952,50.303608,52.304064,46.152384,65.421472,56.843776,63.3776,58.288352,54.328512,52.303648,42.739744,46.759168,56.16544,53.547712,45.519552,50.455584,56.082624,49.677568, 49.44];  
        //5g-mid-band
	const networkthroughput= [ 636.69536, 613.207168, 570.425344, 398.45888, 207.19872, 365.74336, 341.416448, 198.810112, 209.7152, 316.250624, 387.553792, 432.851968, 369.098752, 243.269632, 225.65376, 291.0848, 352.321536, 293.60128, 532.676608, 274.307584, 220.620288, 200.487936, 382.52032, 283.53536, 332.188672, 363.227136, 373.293056, 443.757568, 331.350016, 352.321536, 372.4544, 432.013312, 395.941888, 343.094272, 353.998848, 273.468416, 323.800064, 284.374016, 208.037888, 234.042368, 171.127808, 280.179712, 310.378496, 362.387456, 311.217152, 257.529856, 390.908928, 416.913408, 394.264576, 318.76608, 403.492864, 411.041792, 316.250112, 281.018368, 260.886528, 210.55488, 223.975424, 263.401472, 260.886528, 260.046848, 417.753088, 510.027776, 463.890432, 520.933376, 506.671104, 366.58176, 429.49632, 414.396416, 583.008256, 503.31648, 520.093696, 393.424896, 411.041792, 387.55328, 379.164672, 395.941888, 386.715648, 453.824512, 430.336, 457.179136,
384.198656, 423.624704, 531.836928, 558.682112, 517.576704, 551.970816, 511.705088, 577.974272, 536.870912, 462.21312, 606.49472, 482.34496, 515.059712, 430.333952, 548.614144, 521.773056, 606.49472, 618.242048, 624.951296, 471.441408, 569.585664, 457.179136, 566.23104, 650.9568, 443.756544, 474.796032, 545.25952, 567.07072, 504.99584, 510.025728, 417.75104, 471.441408, 579.653632, 572.940288, 551.968768, 559.521792, 569.585664, 536.870912, 382.521344, 576.299008, 598.945792, 618.242048, 623.271936, 603.979776, 611.528704, 689.545216, 638.373888, 546.93888, 526.802944, 644.243456, 629.98528, 345.612288, 651.79648, 666.054656, 624.951296, 661.020672, 647.602176, 597.270528, 607.3344, 680.316928, 591.396864, 636.694528, 707.158016, 654.311424, 699.609088, 658.505728, 612.368384, 702.128128, 695.414784, 658.509824];

	const i = segmentIndex;
	const netThroughput = i < networkthroughput.length ? networkthroughput[i] : 0;
        if(type === 'video'){
            
            const streamProcessor = streamProcessorDict[streamId][type];
            //console.log("the stream processor data in abrcontroller is", streamProcessor.getRepresentation());
            const currentRepresentation = streamProcessor.getRepresentation();        
            const currentBitrate = currentRepresentation.bitrateInKbit;
            if (lastBitrate !== currentBitrate) {
                bitrateVariation = currentBitrate - lastBitrate;
            }
        	else{
        	    bitrateVariation = 0
        	}
        	const droppedFramesData =  dashMetrics.getCurrentDroppedFrames(type);
        	const droppedFrames = droppedFramesData?.droppedFrames ?? 0;
        	const buffer = dashMetrics.getCurrentBufferLevel(type);
        	const rebuffer = dashMetrics.getCurrentRebuffer(type);
        	const averageThroughput = throughputController.getAverageThroughput(type);
        	const throughput = isNaN(averageThroughput) ? -1 : averageThroughput; 
            
        	const httpRequests = dashMetrics.getHttpRequests('video');
        	const lastRequest = httpRequests[httpRequests.length - 1];
        	const downloadLatency = lastRequest.tresponse.getTime() - lastRequest.trequest.getTime();

            const data = {'index': segmentIndex,
            		    'currentBitrate': currentBitrate,
                        'lastBitrate': lastBitrate,
                        'bitrateVariation': bitrateVariation,
                        'buffer': buffer,
                        'RebufferTime': rebuffer,
                        'throughput': throughput,
                        'droppedFrames': droppedFrames,
                        'protocol': protocol,
                        'latency' : downloadLatency,
                        'networkThroughput': netThroughput,
                        'playbackEnded' : playbackEnded};


            try {
                const rlResponse = await queryRLServerAsync(data);
                const protocolDec = Number(rlResponse);
                lastBitrate = currentBitrate;
		        bitrateVariation = 0;
                protocol = protocolDec === 1 ? 'http3' : 'http2';
                return protocol = protocolDec === 1 ? 'http3' : 'http2';;
            } catch (error) {
                console.error("RL query error:", error);
                return 'http2'; // fallback default
            }
        }
    }
    

    instance = {
        checkPlaybackQuality,
        clearDataForStream,
        getAbandonmentStateFor,
        getInitialBitrateFor,
        getOptimalRepresentationForBitrate,
        getPossibleVoRepresentations,
        getPossibleVoRepresentationsFilteredBySettings,
        getRepresentationByAbsoluteIndex,
        initialize,
        isPlayingAtLowestQuality,
        isPlayingAtTopQuality,
        registerStreamType,
        reset,
        setConfig,
        setPlaybackQuality,
        setWindowResizeEventCalled,
        unRegisterStreamType,
        getNextSegmentProtocol, //newly added
        queryRLServerAsync, //newly added
        getUpdatedProtocol,
        setUpdatedProtocol
    };

    setup();

    return instance;
}

AbrController.__dashjs_factory_name = 'AbrController';
const factory = FactoryMaker.getSingletonFactory(AbrController);
FactoryMaker.updateSingletonFactory(AbrController.__dashjs_factory_name, factory);
export default factory;
