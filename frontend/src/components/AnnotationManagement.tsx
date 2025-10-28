import { FC, useCallback, useEffect, useRef, useState } from 'react';
import { FaPencilAlt } from 'react-icons/fa';
import { LuRefreshCw } from 'react-icons/lu';
import { PiEraser } from 'react-icons/pi';
import { useNavigate, useParams } from 'react-router-dom';
import { Tooltip } from 'react-tooltip';
import { BackButton } from '../components/BackButton';
import { ForwardButton } from '../components/ForwardButton';
import {
  useAddAnnotation,
  useGetElementById,
  useGetNextElementId,
  useStatistics,
} from '../core/api';
import { useAppContext } from '../core/context';
import { ElementOutModel } from '../types';

import { Modal } from 'react-bootstrap';
import { FaMapMarkedAlt } from 'react-icons/fa';
import { MdDisplaySettings } from 'react-icons/md';
import { ActiveLearningManagement } from '../components/ActiveLearningManagement';
import { MulticlassInput } from '../components/MulticlassInput';
import { MultilabelInput } from '../components/MultilabelInput';
import { SelectionManagement } from '../components/SelectionManagement';
import { TagDisplayParameters } from '../components/TagDisplayParameters';
import { TextClassificationPanel } from '../components/TextClassificationPanel';
import { TextSpanPanel } from '../components/TextSpanPanel';

export const AnnotationManagement: FC = () => {
  // parameters
  const { projectName, elementId } = useParams();

  const {
    appContext: {
      currentScheme,
      currentProject: project,
      selectionConfig,
      displayConfig,
      freqRefreshSimpleModel,
      activeSimpleModel,
      history,
      selectionHistory,
      phase,
    },
    setAppContext,
  } = useAppContext();

  const navigate = useNavigate();
  const [element, setElement] = useState<ElementOutModel | null>(null); //state for the current element
  const [nSample, setNSample] = useState<number | null>(null); // specific info
  const [displayComment, setDisplayComment] = useState(false);
  const [comment, setComment] = useState('');
  const [showDisplayConfig, setShowDisplayConfig] = useState<boolean>(false);
  const [showDisplayViz, setShowDisplayViz] = useState<boolean>(false);
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
    activeSimpleModel || null,
  );
  const { getElementById } = useGetElementById(
    projectName || null,
    currentScheme || null,
    activeSimpleModel || null,
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
      getElementById(elementId, phase).then((element) => {
        if (element) setElement(element);
        else {
          navigate(`/projects/${projectName}/tag/noelement`);
          setElement(null);
        }
      });
      reFetchStatistics();
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
      if (res && res.element_id) navigate(`/projects/${projectName}/tag/${res.element_id}`);
      else {
        navigate(`/projects/${projectName}/tag/noelement`);
      }
    });
  };

  const isValidRegex = (pattern: string) => {
    try {
      new RegExp(pattern);
      return true;
    } catch (e) {
      return false;
    }
  };
  const highlightTextRaw = [selectionConfig.filter, ...displayConfig.highlightText.split('\n')];
  const highlightText = highlightTextRaw.filter(
    (text): text is string => typeof text === 'string' && text.trim() !== '',
  );

  // Now filter by valid regex
  const validHighlightText = highlightText.filter(isValidRegex);

  //display switch to test mode
  const isTest = statistics?.test_set_n ? statistics?.test_set_n > 0 : false;
  const isValid = statistics?.valid_set_n ? statistics?.valid_set_n > 0 : false;

  // existing simplemodels
  const availableSimpleModels = project?.simplemodel.available[currentScheme || ''] || [];

  // display active menu
  const [activeMenu, setActiveMenu] = useState<boolean>(false);

  const statisticsDataset = (dataset: string) => {
    if (dataset === 'train') return `${statistics?.train_annotated_n}/${statistics?.train_set_n}`;
    if (dataset === 'valid') return `${statistics?.valid_annotated_n}/${statistics?.valid_set_n}`;
    if (dataset === 'test') return `${statistics?.test_annotated_n}/${statistics?.test_set_n}`;
    return '';
  };

  if (!projectName || !currentScheme) return;

  return (
    <div className="container-fluid">
      <div className="row mt-2">
        {
          // annotation mode
          <div>
            <SelectionManagement />
            {/* <div
              className={`d-flex align-items-center mb-3 ${phase !== 'train' ? 'alert alert-warning' : ''}`}
            > */}
            <div className="text-center my-2">
              {statistics ? (
                <span className="badge text-bg-light currentstatistics">
                  <span className="d-none d-md-inline">Annotated : </span>
                  {statisticsDataset(phase)} ;{' '}
                  <span className="d-none d-md-inline">Selected : </span>
                  {nSample || ''}
                  <Tooltip anchorSelect=".currentstatistics" place="top">
                    statistics for the current scheme
                  </Tooltip>
                </span>
              ) : (
                ''
              )}
              <button className="btn btn-secondary getelement" onClick={refetchElement}>
                <LuRefreshCw size={20} /> <span className="d-none d-md-inline">Get element</span>
                <Tooltip anchorSelect=".getelement" place="top">
                  Get next element with the selection mode
                </Tooltip>
              </button>
            </div>

            {/* <div>
                <button
                  type="button"
                  className={`btn btn-sm ${activeSimpleModel ? 'btn-success' : 'btn-outline-warning'} rounded-pill px-2 py-1`}
                  onClick={() => setActiveMenu(!activeMenu)}
                >
                  active
                </button>
              </div> */}
            {activeMenu && (
              <ActiveLearningManagement
                projectSlug={projectName}
                history={history}
                currentScheme={currentScheme}
                availableSimpleModels={availableSimpleModels}
                setAppContext={setAppContext}
                freqRefreshSimpleModel={freqRefreshSimpleModel}
                activeSimepleModel={activeSimpleModel}
              />
            )}
          </div>
        }
      </div>
      {kindScheme !== 'span' ? (
        <>
          {elementId === 'noelement' && (
            <div className="alert alert-warning text-center">
              <div className="m-2">No element available</div>
              <button className="btn btn-primary" onClick={refetchElement}>
                Get element
              </button>
            </div>
          )}

          {
            // display content
          }

          {!isValidRegex(selectionConfig.filter || '') && (
            <div className="alert alert-danger">Regex not valid</div>
          )}

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
            postAnnotation={postAnnotation}
            labels={availableLabels}
            text={element?.text as string}
            lastTag={lastTag as string}
          />
        </>
      )}
      {elementId !== 'noelement' && (
        <div className="row">
          <div className="d-flex flex-wrap gap-2 justify-content-center">
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
              />
            )}
            {kindScheme == 'multilabel' && (
              <MultilabelInput
                elementId={elementId || 'noelement'}
                postAnnotation={postAnnotation}
                labels={availableLabels}
              />
            )}

            {
              // erase button to remove last annotation
              lastTag && (
                <button
                  className="btn clearannotation"
                  onClick={() => {
                    postAnnotation(null, elementId);
                  }}
                >
                  <PiEraser />
                  <Tooltip anchorSelect=".clearannotation" place="top">
                    Erase current tag
                  </Tooltip>
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
        </div>
      )}
      <div className="d-flex flex-wrap gap-2 justify-content-center">
        <button className="btn addcomment" onClick={() => setDisplayComment(!displayComment)}>
          <FaPencilAlt />
          <Tooltip anchorSelect=".addcomment" place="top">
            Add a commentary
          </Tooltip>
        </button>

        <button
          className="btn displayconfig"
          onClick={() => {
            setShowDisplayConfig(!showDisplayConfig);
          }}
        >
          <MdDisplaySettings />
          <Tooltip anchorSelect=".displayconfig" place="top">
            Display config menu
          </Tooltip>
        </button>
        <button
          className="btn displayviz"
          onClick={() => {
            setShowDisplayViz(!showDisplayConfig);
          }}
        >
          <FaMapMarkedAlt />
          <Tooltip anchorSelect=".displayviz" place="top">
            Display the projection
          </Tooltip>
        </button>
      </div>

      <Modal show={displayComment} onHide={handleCloseComment} id="comment-modal">
        <Modal.Header closeButton>
          <Modal.Title>Add a commentary to the label</Modal.Title>
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
          <button className="btn btn-primary" onClick={handleCloseComment}>
            Save
          </button>
        </Modal.Footer>
      </Modal>
      <Modal show={showDisplayViz} onHide={handleCloseViz} size="xl" id="viz-modal">
        <Modal.Header closeButton>
          <Modal.Title>Projection</Modal.Title>
        </Modal.Header>
        <Modal.Body>TO IMPLEMENT</Modal.Body>
      </Modal>
      <Modal show={showDisplayConfig} onHide={handleCloseConfig} size="xl" id="config-modal">
        <Modal.Header closeButton>
          <Modal.Title>Configuration</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <TagDisplayParameters displayConfig={displayConfig} setAppContext={setAppContext} />
        </Modal.Body>
      </Modal>
    </div>
  );
};
