import { FC, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FaPencilAlt } from 'react-icons/fa';
import { FiRefreshCcw } from 'react-icons/fi';
import { LuRefreshCw } from 'react-icons/lu';
import { PiEraser } from 'react-icons/pi';
import { useNavigate, useParams } from 'react-router-dom';
import { BackButton } from '../components/BackButton';
import { ForwardButton } from '../components/ForwardButton';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useRetrainQuickModel,
  useStatistics,
  useTrainQuickModel,
} from '../core/api';
import { useAppContext } from '../core/context';
import { ElementOutModel } from '../types';

import { Modal } from 'react-bootstrap';
import { FaMapMarkedAlt } from 'react-icons/fa';
import { MdDisplaySettings } from 'react-icons/md';
import { ActiveLearningManagement } from '../components/ActiveLearningManagement';
import { MulticlassInput } from '../components/MulticlassInput';
import { MultilabelInput } from '../components/MultilabelInput';
import { TagDisplayParameters } from '../components/TagDisplayParameters';
import { TextClassificationPanel } from '../components/TextClassificationPanel';
import { TextSpanPanel } from '../components/TextSpanPanel';
import { useNotifications } from '../core/notifications';
import { isValidRegex } from '../core/utils';
import { AnnotationModeForm } from './AnnotationMode';
import { DisplayProjection } from './vizualisation/DisplayProjection';

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
  const [displayComment, setDisplayComment] = useState(false);
  const [comment, setComment] = useState('');
  const [showDisplayConfig, setShowDisplayConfig] = useState<boolean>(false);
  const [showDisplayViz, setShowDisplayViz] = useState<boolean>(false);
  const [settingChanged, setSettingChanged] = useState<boolean>(false);
  const [selectFirstModelTrained, setSelectFirstModelTrained] = useState<boolean>(false);
  const handleCloseViz = () => setShowDisplayViz(false);
  const handleCloseConfig = () => setShowDisplayConfig(false);
  const handleCloseComment = () => setDisplayComment(false);

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
    history,
    phase,
    activeModel || null,
  );
  const { getElementById } = useGetElementById(
    projectName || null,
    currentScheme || null,
    activeModel || null,
  );

  // hooks to manage annotation
  const { addAnnotation } = useAddAnnotation(projectName || null, currentScheme || null, phase);

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

  // hook to clear history
  const actionClearHistory = () => {
    setAppContext((prev) => ({ ...prev, history: [] }));
  };
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
          navigate(`/projects/${projectName}/tag/${res.element_id}`);
        } else {
          navigate(`/projects/${projectName}/tag/noelement`);
          setElement(null);
        }
      });
    } else {
      // only if id changed compared to the previous one (otherwise, a change in phase would trigger a reload)
      if (element?.element_id !== elementId) {
        getElementById(elementId, phase).then((element) => {
          if (element) setElement(element);
          else {
            navigate(`/projects/${projectName}/tag/noelement`);
            setElement(null);
          }
        });
        reFetchStatistics();
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

  // post an annotation
  const postAnnotation = useCallback(
    (label: string | null, elementId?: string) => {
      if (elementId === 'noelement') return; // forbid annotation on noelement
      if (elementId) {
        addAnnotation(elementId, label, comment, selectionHistory[elementId]).then(() =>
          // redirect to next element by redirecting wihout any id
          // thus the getNextElementId query will be dont after the appcontext is reloaded
          {
            setAppContext((prev) => ({ ...prev, history: [...prev.history, elementId] }));
            setComment('');
            navigate(`/projects/${projectName}/tag/`); // got to next element
          },
        );
        // does not do nothing as we remount through navigate reFetchStatistics();
      }
    },
    [setAppContext, addAnnotation, navigate, projectName, comment, selectionHistory],
  );

  const textInFrame = element?.text.slice(0, displayConfig.numberOfTokens * 4) || '';
  const textOutFrame = element?.text.slice(displayConfig.numberOfTokens * 4) || '';

  const lastTag =
    element?.history?.length && element?.history.length > 0
      ? (element?.history[0] as string[])[0]
      : null;

  const refetchElement = () => {
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
  };

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

  const statisticsDataset = useMemo(() => {
    if (phase === 'train') return `${statistics?.train_annotated_n}/${statistics?.train_set_n}`;
    if (phase === 'valid') return `${statistics?.valid_annotated_n}/${statistics?.valid_set_n}`;
    if (phase === 'test') return `${statistics?.test_annotated_n}/${statistics?.test_set_n}`;
    return '';
  }, [phase, statistics]);

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

  // retrain quick model
  const { retrainQuickModel } = useRetrainQuickModel(projectName || null, currentScheme || null);
  const [updatedQuickModel, setUpdatedQuickModel] = useState(false);
  useEffect(() => {
    if (
      !updatedQuickModel &&
      freqRefreshQuickModel &&
      activeModel &&
      history.length > 0 &&
      history.length % freqRefreshQuickModel == 0 &&
      activeModel.type === 'quickmodel'
    ) {
      setUpdatedQuickModel(true);
      retrainQuickModel(activeModel.value);
    }
    if (updatedQuickModel && freqRefreshQuickModel && history.length % freqRefreshQuickModel != 0) {
      setUpdatedQuickModel(false);
    }
  }, [
    freqRefreshQuickModel,
    setUpdatedQuickModel,
    activeModel,
    updatedQuickModel,
    retrainQuickModel,
    history.length,
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

  if (!projectName || !currentScheme) return;

  return (
    <>
      {/**
       * Annotation mode form
       * **/}
      <AnnotationModeForm
        settingChanged={settingChanged}
        setSettingChanged={setSettingChanged}
        refetchElement={refetchElement}
        setActiveMenu={setActiveMenu}
      />
      <div>
        {/* TODO: find a design for this */}
        {statistics ? (
          <span className="explanations">
            Annotated: {statisticsDataset} - Selected: {nSample || 'na'}
          </span>
        ) : (
          'na'
        )}
      </div>

      {/**
       *  ANNOTATION BLOCK
       * */}
      {elementId === 'noelement' ? (
        <div className="alert horizontal center">
          <div>
            No element available
            <button className="btn-primary-action" onClick={refetchElement}>
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
            postAnnotation={postAnnotation}
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
      {/* NOTE: Axel Not too much customisation cause it's gonna be refactored soon */}
      {elementId !== 'noelement' && (
        <div className="horizontal center">
          <BackButton
            projectName={projectName || ''}
            history={history}
            setAppContext={setAppContext}
          />

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

          <button
            className="transparent-background"
            onClick={() => setDisplayComment(!displayComment)}
            title="Add a comment"
          >
            <FaPencilAlt />
          </button>
          {
            // erase button to remove last annotation
            lastTag && (
              <button
                className="transparent-background"
                onClick={() => {
                  postAnnotation(null, elementId);
                }}
                title="Erase current tag"
              >
                <PiEraser />
              </button>
            )
          }
          {elementId && (
            <ForwardButton
              setAppContext={setAppContext}
              elementId={elementId}
              refetchElement={refetchElement}
            />
          )}
        </div>
      )}
      <div className="horizontal center">
        {/* NOTE: Axel Not too much customisation cause it's gonna be refactored soon */}
        <button
          className="transparent-background"
          onClick={() => {
            setShowDisplayConfig(!showDisplayConfig);
          }}
          title="Display config menu"
        >
          <MdDisplaySettings />
        </button>
        <button
          className="transparent-background"
          onClick={() => {
            setShowDisplayViz(!showDisplayConfig);
          }}
          title="Display the projection"
        >
          <FaMapMarkedAlt />
        </button>
        <button
          className="transparent-background"
          onClick={actionClearHistory}
          title="Clear the history"
        >
          <FiRefreshCcw />
        </button>
      </div>

      <Modal show={displayComment} onHide={handleCloseComment} id="comment-modal">
        <Modal.Header closeButton>
          <Modal.Title>Add a comment with your annotation</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <input
            type="text"
            className="form-control"
            placeholder="Comment"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
        </Modal.Body>
        <Modal.Footer>
          <button className="btn-submit" onClick={handleCloseComment}>
            Save
          </button>
        </Modal.Footer>
      </Modal>
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
      <Modal show={showDisplayConfig} onHide={handleCloseConfig} size="xl" id="config-modal">
        <Modal.Header closeButton>
          <Modal.Title>Configuration</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <TagDisplayParameters displayConfig={displayConfig} setAppContext={setAppContext} />
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
