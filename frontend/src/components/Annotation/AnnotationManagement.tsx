import { CSSProperties, FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LuRefreshCw } from 'react-icons/lu';
import { useNavigate, useParams } from 'react-router-dom';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useRetrainQuickModel,
  useStatistics,
  useTrainQuickModel,
} from '../../core/api';
import { useAppContext } from '../../core/context';
import { ElementOutModel } from '../../types';

import classNames from 'classnames';
import { Modal } from 'react-bootstrap';
import { useNotifications } from '../../core/notifications';
import { useAnnotationSessionHistory } from '../../core/useHistory';
import { isValidRegex } from '../../core/utils';
import { ActiveLearningManagement } from '../ActiveLearningManagement';
import { TagDisplayParameters } from '../TagDisplayParameters';
import { DisplayProjection } from '../vizualisation/DisplayProjection';
import { AnnotationHistoryList } from './AnnotationHistoryList';
import { AnnotationModeForm } from './AnnotationMode';
import { MulticlassInput } from './MulticlassInput';
import { MultilabelInput } from './MultilabelInput';
import { TextClassificationPanel } from './TextClassificationPanel';
import { TextSpanPanel } from './TextSpanPanel';

export const AnnotationManagement: FC = () => {
  const { notify } = useNotifications();
  const { projectName, elementId } = useParams();
  const { appContext, setAppContext } = useAppContext();

  const {
    currentScheme,
    currentProject: project,
    selectionConfig,
    displayConfig,
    freqRefreshQuickModel,
    activeModel,
    history,
    selectionHistory,
    phase,
  } = appContext;

  const navigate = useNavigate();
  const [element, setElement] = useState<ElementOutModel | null>(null); //state for the current element
  const [nSample, setNSample] = useState<number | null>(null); // specific info

  const [showDisplayConfig, setShowDisplayConfig] = useState<boolean>(false);
  const [showDisplayViz, setShowDisplayViz] = useState<boolean>(false);
  const [selectFirstModelTrained, setSelectFirstModelTrained] = useState<boolean>(false);
  const handleCloseViz = () => setShowDisplayViz(false);
  const handleCloseConfig = () => setShowDisplayConfig(false);

  // Reinitialize scroll in frame
  const frameRef = useRef<HTMLDivElement>(null);
  const resetScroll = () => {
    if (frameRef.current) {
      frameRef.current.scrollTop = 0;
    }
  };

  // hooks to manage element
  const { getNextElementId } = useGetNextElementId(
    projectName || null,
    currentScheme || null,
    selectionConfig,
    history.map((h) => h.element_id),
    phase,
    activeModel || null,
  );
  const { getElementById } = useGetElementById();

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null, phase);

  //hook to manage history
  const { addElementInAnnotationSessionHistory } = useAnnotationSessionHistory();

  // define parameters for configuration panels
  const availableLabels =
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].labels
      : [];
  const [kindScheme] = useState<string>(
    currentScheme && project && project.schemes.available[currentScheme]
      ? project.schemes.available[currentScheme].kind || 'multiclass'
      : 'multiclass',
  );

  // get statistics to display (TODO : try a way to avoid another request ?)
  const { statistics, reFetchStatistics } = useStatistics(
    projectName || null,
    currentScheme || null,
  );

  // react to URL param change
  useEffect(() => {
    resetScroll();
    if (elementId === 'noelement') {
      return;
    }
    if (elementId === undefined) {
      getNextElementId().then((res) => {
        if (res && res.n_sample) setNSample(res.n_sample);
        if (res && res.element_id) {
          setAppContext((prev) => ({
            ...prev,
            selectionHistory: {
              ...prev.selectionHistory,
              [res.element_id]: JSON.stringify(selectionConfig),
            },
          }));
          // redirect to the next element page replacing history
          navigate(`/projects/${projectName}/tag/${res.element_id}`, { replace: true });
        } else {
          navigate(`/projects/${projectName}/tag/noelement`);
          setElement(null);
        }
      });
    } else {
      // only if id changed compared to the previous one (otherwise, a change in phase would trigger a reload)
      if (element?.element_id !== elementId) {
        getElementById(elementId, phase)
          .then((element) => {
            if (element) setElement(element);
            else {
              navigate(`/projects/${projectName}/tag/noelement`);
              setElement(null);
            }
          })
          .finally(() => {
            //info: get statistics call returns often outdated data
            reFetchStatistics();
          });
      }
    }
  }, [
    elementId,
    getNextElementId,
    getElementById,
    navigate,
    phase,
    projectName,
    reFetchStatistics,
    selectionConfig,
    setAppContext,
    notify,
    element,
  ]);

  const postAnnotation = useCallback(
    async (label: string | null, elementId: string, comment?: string) => {
      if (elementId === 'noelement') return; // forbid annotation on noelement
      if (elementId) {
        await addAnnotation(elementId, label, comment || null, selectionHistory[elementId]);
        const newElement = await getElementById(elementId, phase);
        if (newElement) {
          addElementInAnnotationSessionHistory(elementId, newElement.text, label, comment);
          setElement(newElement);
          // wait for 500ms before fetch new element to see new button state
          setTimeout(() => {
            navigate(`/projects/${projectName}/tag/`);
          }, 200);
        }
        // does not do nothing as we remount through navigate reFetchStatistics();
      }
    },
    [
      addAnnotation,
      selectionHistory,
      projectName,
      navigate,
      getElementById,
      setElement,
      phase,
      addElementInAnnotationSessionHistory,
    ],
  );

  const textInFrame = element?.text.slice(0, displayConfig.numberOfTokens * 4) || '';
  const textOutFrame = element?.text.slice(displayConfig.numberOfTokens * 4) || '';

  const lastTag = element?.history && element.history.length > 0 ? element.history[0].label : null;

  const fetchNextElement = useCallback(() => {
    getNextElementId().then((res) => {
      if (res && res.n_sample) setNSample(res.n_sample);
      if (res && res.element_id) {
        if (res.element_id === elementId) {
          notify({
            type: 'warning',
            message:
              'Next selected element is the same as the current one. Try changing selection settings.',
          });
        }
        navigate(`/projects/${projectName}/tag/${res.element_id}`);
      } else {
        navigate(`/projects/${projectName}/tag/noelement`);
      }
    });
  }, [getNextElementId, notify, setNSample, navigate, projectName, elementId]);

  const highlightTextRaw = [selectionConfig.filter, ...displayConfig.highlightText.split('\n')];
  const highlightText = highlightTextRaw.filter(
    (text): text is string => typeof text === 'string' && text.trim() !== '',
  );

  // Now filter by valid regex
  const validHighlightText = highlightText.filter(isValidRegex);

  // existing models
  const availableQuickModels = useMemo(
    () => project?.quickmodel.available[currentScheme || ''] || [],
    [project?.quickmodel, currentScheme],
  );
  const availableBertModels = useMemo(
    () => project?.languagemodels.available[currentScheme || ''] || {},
    [project?.languagemodels, currentScheme],
  );
  const availableBertModelsWithPrediction = Object.entries(availableBertModels || {})
    .filter(([_, v]) => v && v.predicted)
    .map(([k, _]) => k);
  //
  // TODO only keep those with prediction
  const groupedModels = [
    {
      label: 'Quick Models',
      options: (availableQuickModels ?? [])
        .filter((e) => e?.name) // <-- protect against undefined/missing name
        .map((e) => ({
          value: e.name,
          label: e.name,
          type: 'quickmodel',
        })),
    },
    {
      label: 'Language Models',
      options: (availableBertModelsWithPrediction ?? [])
        .filter((e) => e) // <-- ensure non-null
        .map((e) => ({
          value: e,
          label: e,
          type: 'languagemodel',
        })),
    },
  ];

  // display active menu
  const [activeMenu, setActiveMenu] = useState<boolean>(false);

  // train a quick model
  const { trainQuickModel } = useTrainQuickModel(projectName || null, currentScheme || null);
  const startTrainQuickModel = () => {
    // default quickmodel
    const availableFeatures = project?.features.available ? project?.features.available : [];
    if (availableFeatures.length === 0) {
      setActiveMenu(false);
      notify({
        type: 'warning',
        message: 'No features available for quickmodel',
      });
    }
    const formData = {
      name: 'basic-quickmodel',
      model: 'logistic-l1',
      scheme: currentScheme || '',
      params: {
        costLogL2: 1,
        costLogL1: 1,
        n_neighbors: 3,
        alpha: 1,
        n_estimators: 500,
        max_features: null,
      },
      dichotomize: null,
      features: availableFeatures,
      cv10: false,
      standardize: false,
      balance_classes: false,
    };
    trainQuickModel(formData);
    setActiveMenu(false);
    setSelectFirstModelTrained(true);
  };

  // fastrack active learning model
  useEffect(() => {
    if (selectFirstModelTrained && availableQuickModels.length > 0) {
      // select the first trained model
      setAppContext((prev) => ({
        ...prev,
        activeModel: {
          type: 'quickmodel',
          value: availableQuickModels[0].name,
          label: availableQuickModels[0].name,
        },
        selectionConfig: { ...prev.selectionConfig, mode: 'active' },
      }));
    }
  }, [availableQuickModels, selectFirstModelTrained, setAppContext]);

  // retrain quick model (only for )
  const { retrainQuickModel } = useRetrainQuickModel(projectName || null, currentScheme || null);
  const [updatedQuickModel, setUpdatedQuickModel] = useState(false);
  useEffect(() => {
    // only the training points for the current phase
    const trainHistory = history.filter(
      (hp) => hp.dataset === 'train' && hp.project_slug === projectName,
    );
    if (
      !updatedQuickModel &&
      freqRefreshQuickModel &&
      activeModel &&
      trainHistory.length > 0 &&
      trainHistory.length % freqRefreshQuickModel == 0 &&
      activeModel.type === 'quickmodel'
    ) {
      setUpdatedQuickModel(true);
      retrainQuickModel(activeModel.value);
    }
    if (
      updatedQuickModel &&
      freqRefreshQuickModel &&
      trainHistory.length % freqRefreshQuickModel != 0
    ) {
      setUpdatedQuickModel(false);
    }
  }, [
    freqRefreshQuickModel,
    setUpdatedQuickModel,
    activeModel,
    updatedQuickModel,
    retrainQuickModel,
    history.length,
    projectName,
    history,
  ]);

  // deactivate active model if it has been removed from available models
  useEffect(() => {
    if (
      activeModel &&
      !availableQuickModels.find((model) => model.name === activeModel.value) &&
      activeModel.type === 'quickmodel'
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
    }
    if (
      activeModel &&
      !Object.keys(availableBertModels).includes(activeModel.value) &&
      activeModel.type === 'languagemodel'
    ) {
      setAppContext((prev) => ({ ...prev, activeModel: null }));
    }
  }, [availableQuickModels, activeModel, setAppContext, availableBertModels]);

  /**
   * Update element if selectionConfig changed :
   * - refetch if active model is activated
   * - getNextElement if in noelement page
   */
  const refetchElement = useCallback(async () => {
    if (elementId) {
      const newElement = await getElementById(elementId, phase);
      if (newElement) setElement(newElement);
    }
  }, [setElement, getElementById, elementId, phase]);

  useEffect(() => {
    refetchElement();
    // disabling echaustive deps as we only want to track phase to avoid unnecessary refetch
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeModel]);

  useEffect(() => {
    // fetch next element in the new phase
    // only if there is one current element to avoid triggering fetchnext at page load
    if (element !== null) {
      fetchNextElement();
    }
    // disabling echaustive deps as we only want to track phase to avoid unnecessary fetchNext
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  useEffect(() => {
    if (elementId === 'noelement') {
      fetchNextElement();
    }
    // disabling echaustive deps as we only want to track phase to avoid unnecessary fetchNext
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionConfig]);

  if (!projectName || !currentScheme) return;

  return (
    <>
      {/**
       * Annotation mode form
       **/}
      <AnnotationModeForm
        fetchNextElement={fetchNextElement}
        setActiveMenu={setActiveMenu}
        setShowDisplayViz={setShowDisplayViz}
        setShowDisplayConfig={setShowDisplayConfig}
        nSample={nSample}
        statistics={statistics}
      />

      {/**
       * ANNOTATION BLOCK
       **/}
      <div
        className={classNames(
          'annotation-block',
          (displayConfig.forceOneColumnLayout || kindScheme == 'multilabel') &&
            'force-one-column-layout',
        )} // add class to force bottom if settings OR multiclass label
        style={
          {
            '--text-width': `${displayConfig.textFrameWidth}%`,
          } as CSSProperties
        }
      >
        {elementId === 'noelement' ? (
          <div className="alert horizontal center">
            <div>
              No element available
              <button className="btn-primary-action" onClick={fetchNextElement}>
                <LuRefreshCw size={20} /> Get element
              </button>
            </div>
          </div>
        ) : kindScheme !== 'span' ? (
          <>
            <TextClassificationPanel
              element={element as ElementOutModel}
              displayConfig={displayConfig}
              textInFrame={textInFrame}
              textOutFrame={textOutFrame}
              validHighlightText={validHighlightText}
              elementId={elementId as string}
              lastTag={lastTag as string}
              phase={phase}
              frameRef={frameRef as unknown as HTMLDivElement}
            />
          </>
        ) : (
          <>
            <TextSpanPanel
              elementId={elementId || 'noelement'}
              displayConfig={displayConfig}
              postAnnotation={postAnnotation}
              labels={availableLabels}
              text={element?.text as string}
              lastTag={lastTag as string}
            />
          </>
        )}

        {elementId !== 'noelement' && (
          <>
            {kindScheme == 'multiclass' && (
              <MulticlassInput
                elementId={elementId || 'noelement'}
                postAnnotation={postAnnotation}
                labels={availableLabels}
                phase={phase}
                element={element as ElementOutModel}
              />
            )}
            {kindScheme == 'multilabel' && (
              <MultilabelInput
                elementId={elementId || 'noelement'}
                postAnnotation={postAnnotation}
                labels={availableLabels}
              />
            )}
          </>
        )}
      </div>

      <div>
        {displayConfig.displayHistory ? (
          <AnnotationHistoryList />
        ) : (
          <span
            style={{ cursor: 'pointer', color: 'gray' }}
            onClick={(_) => {
              setAppContext((prev) => ({
                ...prev,
                displayConfig: {
                  ...displayConfig,
                  displayHistory: !displayConfig.displayHistory,
                },
              }));
            }}
          >
            History hidden - click to show
          </span>
        )}
      </div>

      <Modal show={showDisplayViz} onHide={handleCloseViz} size="xl" id="viz-modal">
        <Modal.Header closeButton>
          <Modal.Title>Current projection</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div className="horizontal center" style={{ overflowY: 'scroll' }}>
            <DisplayProjection
              projectName={projectName}
              currentScheme={currentScheme}
              elementId={elementId}
            />
          </div>
        </Modal.Body>
      </Modal>
      <Modal show={showDisplayConfig} onHide={handleCloseConfig} size="sm" id="config-modal">
        <Modal.Header closeButton>
          <Modal.Title>Configuration</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <TagDisplayParameters />
        </Modal.Body>
      </Modal>
      <Modal show={activeMenu} onHide={() => setActiveMenu(false)} id="active-modal" size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Configure active learning</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {availableQuickModels.length > 0 ? (
            <ActiveLearningManagement
              availableModels={groupedModels}
              setAppContext={setAppContext}
              freqRefreshQuickModel={freqRefreshQuickModel}
              activeModel={activeModel}
              projectName={projectName}
              currentScheme={currentScheme}
            />
          ) : (
            <>
              <div className="horizontal center">
                No quick model currently available. Go to model tab or
              </div>
              <button className="btn-submit" onClick={startTrainQuickModel}>
                Train a default quick model
              </button>
            </>
          )}
        </Modal.Body>
      </Modal>
    </>
  );
};
