import cx from 'classnames';
import { FC, useState } from 'react';
import { Modal } from 'react-bootstrap';
import { DisplayTrainingProcesses } from '../components/DisplayTrainingProcesses';
import { ModelCreationForm } from '../components/forms/ModelCreationForm';
import { ModelParametersTab } from '../components/ModelParametersTab';
import { LossChart } from '../components/vizualisation/lossChart';
import { ModelDescriptionModel, ProjectStateModel } from '../types';
import { DisplayScores } from './DisplayScores';
import { ModelsPillDisplay } from './ModelsPillDisplay';
import { ValidateButtons } from './validateButton';

import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { MdDriveFileRenameOutline } from 'react-icons/md';
import { useDeleteBertModel, useModelInformations, useRenameBertModel } from '../core/api';
import { useNotifications } from '../core/notifications';
import { sortDatesAsStrings } from '../core/utils';
import { MLStatisticsModel } from '../types';

interface renameModel {
  new_name: string;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

interface BertModelManagementProps {
  projectSlug: string | null;
  isComputing: boolean;
  project: ProjectStateModel | null;
  currentScheme: string | null;
  availableBertModels: { [key: string]: ModelDescriptionModel };
}

export const BertModelManagement: FC<BertModelManagementProps> = ({
  projectSlug,
  currentScheme,
  project,
  availableBertModels,
  isComputing,
}) => {
  const { notify } = useNotifications();

  const [showParameters, setShowParameters] = useState(false);
  const [showRename, setShowRename] = useState(false);

  // current model and automatic selection

  const [currentBertModel, setCurrentBertModel] = useState<string | null>(
    availableBertModels ? Object.keys(availableBertModels)[0] : null,
  );
  const { deleteBertModel } = useDeleteBertModel(projectSlug || null);

  // get model information from api
  const { model } = useModelInformations(
    projectSlug || null,
    currentBertModel || null,
    'bert',
    isComputing,
  );

  // form to rename
  const { renameBertModel } = useRenameBertModel(projectSlug || null);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentBertModel) {
      await renameBertModel(currentBertModel, data.new_name);
      resetRename();
      setShowRename(false);
    } else notify({ type: 'error', message: 'New name is void' });
  };

  const loss = model?.loss ? (model?.loss as unknown as LossData) : null;

  const [displayNewBertModel, setDisplayNewBertModel] = useState(false);

  return (
    <div>
      <ModelsPillDisplay
        modelNames={Object.values(availableBertModels)
          .sort((bertModelA, bertModelB) =>
            sortDatesAsStrings(bertModelA?.time, bertModelB?.time, true),
          )
          .map((model) => model.name)}
        currentModelName={currentBertModel}
        setCurrentModelName={setCurrentBertModel}
        deleteModelFunction={deleteBertModel}
      >
        <button
          onClick={() => setDisplayNewBertModel(true)}
          className={cx('model-pill ', isComputing ? 'disabled' : '')}
          id="create-new"
        >
          <FaPlusCircle size={20} /> Create new model
        </button>
      </ModelsPillDisplay>
      {isComputing && (
        <DisplayTrainingProcesses
          projectSlug={projectSlug || null}
          processes={project?.languagemodels.training}
          displayStopButton={isComputing}
        />
      )}
      {currentBertModel && (
        <div>
          {model && (
            <div>
              <div className="d-flex my-4">
                <button
                  className="btn btn-outline-secondary btn-sm me-2 d-flex align-items-center"
                  onClick={() => setShowParameters(true)}
                >
                  <FaGear size={18} className="me-1" />
                  Parameters
                </button>

                <button
                  className="btn btn-outline-secondary btn-sm me-2 d-flex align-items-center"
                  onClick={() => setShowRename(true)}
                >
                  <MdDriveFileRenameOutline size={18} className="me-1" />
                  Rename
                </button>
                <ValidateButtons
                  projectSlug={projectSlug}
                  modelName={currentBertModel}
                  kind="bert"
                  currentScheme={currentScheme}
                  id="compute-prediction"
                  buttonLabel="Compute predictions"
                />
              </div>

              <DisplayScores
                title={'Validation scores from the training data (internal validation)'}
                scores={model.scores.internalvalid_scores as MLStatisticsModel}
                modelName={currentBertModel}
                projectSlug={projectSlug}
              />

              <div className="mt-2">
                <LossChart loss={loss} />
              </div>
            </div>
          )}
        </div>
      )}

      <Modal
        show={displayNewBertModel}
        id="createmodel-modal"
        size="xl"
        onHide={() => setDisplayNewBertModel(false)}
      >
        <Modal.Header closeButton>
          <Modal.Title>Train a new BERT model</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {/* <div className="explanations">
            The model will be trained on annotated data. A good practice is to have at least 50
            annotated elements. You can exclude elements with specific labels.{' '}
            <a className="problems m-2">
              <FaTools />
              <Tooltip anchorSelect=".problems" place="top">
                If the model doesn't train, the reason can be the limit of available GPU. Please try
                latter. If the problem persists, contact us.
              </Tooltip>
            </a>
          </div> */}
          <ModelCreationForm
            projectSlug={projectSlug || null}
            currentScheme={currentScheme || null}
            project={project || null}
            isComputing={isComputing}
            setStatusDisplay={setDisplayNewBertModel}
          />
        </Modal.Body>
      </Modal>
      <Modal show={showParameters} id="parameters-modal" onHide={() => setShowParameters(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Parameters of {currentBertModel}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <ModelParametersTab params={model?.params as Record<string, unknown>} />
        </Modal.Body>
      </Modal>
      <Modal show={showRename} id="rename-modal" onHide={() => setShowRename(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Rename {currentBertModel}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <form onSubmit={handleSubmitRename(onSubmitRename)}>
            <input
              id="new_name"
              className="form-control me-2 mt-2"
              type="text"
              placeholder="New name of the model"
              {...registerRename('new_name')}
            />
            <button className="btn btn-primary me-2 mt-2">Rename</button>
          </form>
        </Modal.Body>
      </Modal>
    </div>
  );
};
