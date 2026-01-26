import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { useDeleteFeature, useGetFeatureInfo } from '../core/api';
import { useAppContext } from '../core/context';
import { sortDatesAsStrings } from '../core/utils';
import { FeatureDescriptionModelOut } from '../types';
import { ButtonNewFeature } from './ButtonNewFeature';
import { ModelParametersTab } from './ModelParametersTab';
import { ModelsPillDisplay } from './ModelsPillDisplay';

export default function SimpleTable(data: FeatureDescriptionModelOut) {
  return (
    <ModelParametersTab
      params={{
        Name: data.name,
        User: data.user,
        Time: data.time,
        Kind: data.kind,
        ...data.parameters,
      }}
    />
  );
}

export const FeaturesManagement: FC = () => {
  const { projectName } = useParams();

  // get element from the state
  const {
    appContext: { currentProject: project, isComputing },
  } = useAppContext();

  // API calls
  const { featuresInfo } = useGetFeatureInfo(projectName || null, project);
  const deleteFeature = useDeleteFeature(projectName || null);

  // show the menu
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const deleteSelectedFeature = async (element: string) => {
    await deleteFeature(element);
    setSelectedFeature(null);
  };

  if (!project) {
    return <div>No project selected</div>;
  }
  return (
    <div className="row">
      <ModelsPillDisplay
        modelNames={Object.values(featuresInfo || {})
          .sort((featureA, featureB) => sortDatesAsStrings(featureA?.time, featureB?.time, true))
          .map((feature) => (feature && feature.name ? feature.name : ''))}
        currentModelName={selectedFeature}
        setCurrentModelName={setSelectedFeature}
        deleteModelFunction={deleteSelectedFeature}
      >
        <ButtonNewFeature projectSlug={projectName || ''} />
      </ModelsPillDisplay>
      {/* Display computing features */}
      {Object.entries(project?.features.training).map(([key, element]) => (
        <div className="card text-bg-light m-3 bg-warning w-75" key={key}>
          <div className="d-flex m-2 align-items-center">
            Currently computing {element ? element.name : ''}
            {element?.progress ? ` (${element.progress}%)` : ''}
          </div>
        </div>
      ))}
      {featuresInfo &&
        selectedFeature &&
        SimpleTable(featuresInfo[selectedFeature] as FeatureDescriptionModelOut)}
    </div>
  );
};
